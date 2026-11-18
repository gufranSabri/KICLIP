#!/usr/bin/env python3
# Picks novel-split indices where DCGD (the last --checkpoints entry) predicts correctly and every
# other checkpoint predicts wrong -- the clips where the full method's vision features alone flip the
# decision, for scripts/figures/eigencam.py to visualize.
#
#     python scripts/figures/select_eigencam_indices.py --split b2n/hmdb51/novel \
#         --checkpoints runs/main/hmdb51_froster runs/main/hmdb51_dd runs/main/hmdb51_cgas_noki runs/main/hmdb51_cgas \
#         --num-indices 10

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scripts.analysis.common import load_checkpoint  # noqa: E402
from src.config import load_config, parse_overrides  # noqa: E402
from src.data.dataset import VideoDataset  # noqa: E402
from src.data.prompts import tokenize  # noqa: E402
from src.models.kiclip import class_features, clip_logits  # noqa: E402
from src.utils.seed import pick_device, seed_everything  # noqa: E402


def predict(spec: str, cfg, dataset: VideoDataset, tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    model, _ = load_checkpoint(spec, cfg, device)
    with torch.no_grad():
        classes = class_features(model.model, tokens.to(device))
        preds = []
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        for video, _ in loader:
            b, v = video.shape[:2]
            frames, *_ = model.encode_video(video.to(device).flatten(0, -4))
            logits = clip_logits(model.model, frames, classes, model.num_frames).view(b, v, -1).sum(1)
            preds.append(logits.argmax(-1).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="last entry is treated as the full method")
    ap.add_argument("--num-indices", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    seed_everything(cfg.seed)
    device = pick_device()

    dataset = VideoDataset(cfg, args.split, train=False, views=1)
    labels = torch.tensor(dataset.labels)
    tokens = tokenize(dataset.classnames)

    preds = [predict(spec, cfg, dataset, tokens, device) for spec in args.checkpoints]
    correct = [p == labels for p in preds]

    full_right = correct[-1]
    others_wrong = torch.stack(correct[:-1]).logical_not().all(0)
    candidates = (full_right & others_wrong).nonzero().flatten().tolist()

    if len(candidates) < args.num_indices:
        print(f"  only {len(candidates)} indices have DCGD right and every baseline wrong "
              f"(wanted {args.num_indices}); printing all of them", file=sys.stderr)
    chosen = candidates[: args.num_indices]
    print(" ".join(str(i) for i in chosen))


if __name__ == "__main__":
    main()
