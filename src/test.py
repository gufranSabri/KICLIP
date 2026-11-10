from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.data.dataset import VideoDataset
from src.data.prompts import tokenize
from src.models.clip import CLIP, has_injection
from src.models.kiclip import KICLIP, class_features, clip_logits, frame_features
from src.utils import dist
from src.utils.logger import get_logger
from src.utils.metrics import format_metrics
from src.utils.progress import progress
from src.utils.seed import seed_everything

logger = get_logger(__name__)


@torch.no_grad()
def evaluate(model: KICLIP, dataset: VideoDataset, tokens: torch.Tensor, cfg: Config, device: torch.device,
             desc: str = "eval") -> dict:
    rank, world = dist.rank_world()
    was_training = model.training
    model.eval()
    amp = cfg.test.amp and device.type == "cuda"

    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        classes = class_features(model.model, tokens.to(device))  # all prompts, once per split
    loader = DataLoader(
        Subset(dataset, range(rank, len(dataset), world)),
        batch_size=cfg.test.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    k = min(5, classes.shape[0])
    correct1 = correctk = n = 0
    for video, labels in progress(loader, desc=desc, leave=False, ncols=100, disable=rank != 0):
        video, labels = video.to(device, non_blocking=True), labels.to(device)
        b, v = video.shape[:2]
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            frames = frame_features(model.model, video)
            logits = clip_logits(model.model, frames, classes, model.num_frames).view(b, v, -1).sum(1)
        top = logits.topk(k, dim=-1).indices
        correct1 += (top[:, 0] == labels).sum().item()
        correctk += (top == labels[:, None]).any(-1).sum().item()
        n += b

    correct1, correctk, n = dist.all_sum([correct1, correctk, n], device)
    model.train(was_training)
    return {"top1": 100 * correct1 / n, "top5": 100 * correctk / n, "n": int(n)}


def load_weights(path: Path) -> tuple[dict[str, torch.Tensor], dict | None]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt.get("model_state", {})  # ours, or an original FROSTER / KICLIP release
    weights = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    if not weights:
        raise ValueError(f"{path} holds no 'model.*' CLIP weights")
    return weights, ckpt.get("config")


def _check_config(ckpt_cfg: dict | None, cfg: Config) -> None:
    if ckpt_cfg is None:
        logger.warning("checkpoint carries no config; evaluating with this config's model.temporal / data.num_frames")
        return
    for section, key in (("model", "temporal"), ("data", "num_frames"), ("data", "sampling_rate")):
        trained, now = ckpt_cfg[section].get(key), getattr(getattr(cfg, section), key)
        if trained != now:
            logger.warning(
                "%s.%s mismatch: checkpoint was trained with %r, this config has %r. "
                "Pass --set %s.%s=%r to match the checkpoint.",
                section, key, trained, now, section, key, trained,
            )


def test(cfg: Config, checkpoint: str | Path, splits: list[str] | None = None) -> dict:
    rank, world, device = dist.setup()
    if rank:
        logging.getLogger().setLevel(logging.WARNING)
    seed_everything(cfg.train.seed)

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    splits = list(splits or cfg.data.test)
    for split in splits:
        cfg.check_split(split, "--splits")

    weights, ckpt_cfg = load_weights(path)
    _check_config(ckpt_cfg, cfg)
    if has_injection(weights) != cfg.model.inject:
        raise ValueError(
            f"{path} {'has' if has_injection(weights) else 'lacks'} knowledge-injection blocks but model.inject="
            f"{cfg.model.inject}; pass --set model.inject={str(has_injection(weights)).lower()}"
        )
    clip = CLIP.from_state_dict(
        weights, temporal=cfg.model.temporal, num_frames=cfg.data.num_frames, inject=cfg.model.inject
    )
    model = KICLIP(clip, cfg.data.num_frames, distill=False).to(device)

    logger.info("checkpoint   : %s", path)
    logger.info("model        : temporal %s | injection %s", cfg.model.temporal, cfg.model.inject)
    logger.info("device       : %s x %d", device, world)
    logger.info("views        : %d clips x %d frames, crop %d", cfg.test.views, cfg.data.num_frames, cfg.data.crop_size)

    results = {}
    for split in splits:
        dataset = VideoDataset(cfg, split, train=False, views=cfg.test.views)
        metrics = evaluate(model, dataset, tokenize(dataset.classnames), cfg, device, desc=split)
        logger.info("TEST %-22s | %s", split, format_metrics(metrics))
        if rank == 0:
            dest = path.parent / f"test_{split.replace('/', '_')}.json"
            dest.write_text(json.dumps({
                "split": split,
                "checkpoint": str(path),
                "classes": len(dataset.classnames),
                "views": cfg.test.views,
                "metrics": metrics,
            }, indent=2))
            logger.info("wrote %s", dest)
        results[split] = metrics

    dist.cleanup()
    return results
