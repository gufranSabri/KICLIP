# per-class breakdown on a test split: dumps predictions, per-class accuracy and a confusion matrix to JSON.
# read-only wrt training; run after test.py has produced an averaged.pt. Needs a GPU allocation (encode_video on CPU
# is impractical for a full split -- see memory killarney-login-node-limits).

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_config, parse_overrides
from src.data.dataset import VideoDataset
from src.data.prompts import tokenize
from src.models.kiclip import build_kiclip, class_features, clip_logits
from src.test import load_weights


@torch.no_grad()
def predict(model, dataset: VideoDataset, tokens: torch.Tensor, cfg, device) -> tuple[list[int], list[int]]:
    model.eval()
    classes = class_features(model.model, tokens.to(device))
    loader = DataLoader(dataset, batch_size=cfg.test.batch_size, num_workers=cfg.data.num_workers)
    preds, labels_out = [], []
    for video, labels in loader:
        video = video.to(device)
        b, v = video.shape[:2]
        frames = model.encode_video(video)[0]
        logits = clip_logits(model.model, frames, classes, model.num_frames).view(b, v, -1).sum(1)
        preds += logits.argmax(-1).cpu().tolist()
        labels_out += labels.tolist()
    return preds, labels_out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state, _ = load_weights(Path(args.checkpoint))
    weights = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    model = build_kiclip(cfg, weights).to(device)
    model.load_state_dict(state, strict=False)

    dataset = VideoDataset(cfg, args.split, train=False, views=cfg.test.views)
    tokens = tokenize(dataset.classnames)
    preds, labels = predict(model, dataset, tokens, cfg, device)

    classnames = [d.split(":")[0] for d in dataset.classnames]
    per_class: dict[str, dict] = {}
    confused = Counter()
    for name in classnames:
        per_class[name] = {"n": 0, "correct": 0}
    for pred, label in zip(preds, labels):
        c = classnames[label]
        per_class[c]["n"] += 1
        if pred == label:
            per_class[c]["correct"] += 1
        else:
            confused[(c, classnames[pred])] += 1
    for name, d in per_class.items():
        d["acc"] = 100 * d["correct"] / d["n"] if d["n"] else 0.0

    out = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "per_class": per_class,
        "confusions": [{"true": t, "pred": p, "n": n} for (t, p), n in confused.most_common(30)],
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
