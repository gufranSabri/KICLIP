from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.data.dataset import VideoDataset
from src.data.prompts import tokenize
from src.models.clip import has_injection
from src.models.kiclip import KICLIP, build_kiclip, class_features, clip_logits
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
    field_on = model.field_source is not None
    field_correct1 = field_correctk = 0
    for video, labels in progress(loader, desc=desc, leave=False, ncols=100, disable=rank != 0):
        video, labels = video.to(device, non_blocking=True), labels.to(device)
        b, v = video.shape[:2]
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            frames, _, _, _, field_raw = model.encode_video(video)  # frames already carries the field correction
            logits = clip_logits(model.model, frames, classes, model.num_frames).view(b, v, -1).sum(1)
        top = logits.topk(k, dim=-1).indices
        correct1 += (top[:, 0] == labels).sum().item()
        correctk += (top == labels[:, None]).any(-1).sum().item()
        n += b
        if field_on:
            # the field's OWN classifier, standalone (not mixed with the main head); open-vocabulary (same
            # embed_dim as the main head), so this works on any split's class count, including novel.
            # F.normalize runs in fp32 under autocast regardless of field_raw's dtype; match classes' dtype
            # before the matmul in clip_logits, which autocast does NOT upcast for us here
            field_normed = F.normalize(field_raw, dim=-1).to(classes.dtype)
            field_logits = clip_logits(model.model, field_normed, classes, model.num_frames).view(b, v, -1).sum(1)
            field_top = field_logits.topk(k, dim=-1).indices
            field_correct1 += (field_top[:, 0] == labels).sum().item()
            field_correctk += (field_top == labels[:, None]).any(-1).sum().item()

    correct1, correctk, n = dist.all_sum([correct1, correctk, n], device)
    metrics = {"top1": 100 * correct1 / n, "top5": 100 * correctk / n, "n": int(n)}
    if field_on:
        field_correct1, field_correctk = dist.all_sum([field_correct1, field_correctk], device)
        metrics["field_top1"] = 100 * field_correct1 / n
        metrics["field_top5"] = 100 * field_correctk / n
    model.train(was_training)
    return metrics


def load_weights(path: Path) -> tuple[dict[str, torch.Tensor], dict | None]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt.get("model_state", {})  # ours, or an original FROSTER / KICLIP release
    if not any(k.startswith("model.") for k in state):
        raise ValueError(f"{path} holds no 'model.*' CLIP weights")
    return state, ckpt.get("config")


def injection(model: dict) -> str | None:
    # a saved model config's injection source, None when off; configs from before inject_source used blocks
    return model.get("inject_source", "blocks") if model["inject"] else None


def _check_config(ckpt_cfg: dict | None, cfg: Config) -> None:
    if ckpt_cfg is None:
        logger.warning("checkpoint carries no config; evaluating with this config's model.temporal / data.num_frames")
        return
    trained, now = injection(ckpt_cfg["model"]), injection(cfg.to_dict()["model"])
    if trained != now:
        raise ValueError(
            f"checkpoint was trained with injection from {trained or 'nothing'}, this config injects from "
            f"{now or 'nothing'}; stack the matching configs/methods/*.yaml"
        )
    trained_motion = ckpt_cfg["model"].get("motion", False)
    if trained_motion != cfg.model.motion:
        raise ValueError(
            f"checkpoint was trained with model.motion={trained_motion}, this config has model.motion={cfg.model.motion}; "
            "stack the matching configs/methods/motion.yaml"
        )
    if trained_motion:
        for key in ("motion_source", "motion_site", "motion_blocks", "motion_bottleneck"):
            trained, now = ckpt_cfg["model"].get(key), getattr(cfg.model, key)
            if trained != now:
                raise ValueError(
                    f"checkpoint was trained with model.{key}={trained!r}, this config has model.{key}={now!r}; "
                    f"pass --set model.{key}={trained!r}"
                )
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
    seed_everything(cfg.seed)

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    splits = list(splits or cfg.data.test)
    for split in splits:
        cfg.check_split(split, "--splits")

    state, ckpt_cfg = load_weights(path)
    _check_config(ckpt_cfg, cfg)
    weights = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    source = injection(cfg.to_dict()["model"])
    if has_injection(weights) != (source == "blocks"):
        raise ValueError(
            f"{path} has frozen injection blocks; test it with configs/methods/kiclip_blocks.yaml"
            if has_injection(weights) else f"{path} lacks the frozen injection blocks model.inject_source=blocks needs"
        )
    model = build_kiclip(cfg, weights).to(device)
    # build_kiclip only loads the inner CLIP submodule from `weights`; load the rest of the checkpoint (field_embed,
    # aux, projector_v/t) on top -- strict=False since a fresh model may have modules this checkpoint predates
    model.load_state_dict(state, strict=False)

    logger.info("checkpoint   : %s", path)
    logger.info("model        : temporal %s | injection from %s", cfg.model.temporal, source or "nothing")
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
