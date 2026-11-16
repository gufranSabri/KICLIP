#!/usr/bin/env python3
# Dumps what report.py analyses: per-frame embeddings and CGAS inputs, per-block CLS / patch-mean / frozen-stream
# tokens and class text features, for zero-shot CLIP and every run, over data.train + data.test. One decode per batch.
#
#     python scripts/analysis/dump.py --config configs/b2n/default/hmdb51.yaml \
#         --runs runs/froster runs/dd runs/cgas

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import Config, ModelConfig, load_config, parse_overrides  # noqa: E402
from src.models.clip import CLIP, pretrained_state  # noqa: E402
from src.models.cgas import pool_segments  # noqa: E402
from src.models.kiclip import KICLIP, build_kiclip, class_features  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.seed import pick_device, seed_everything  # noqa: E402

CLIP_NAME = "clip"
RUNS = ("runs/froster", "runs/dd", "runs/cgas")


class Recorder:
    # forward hooks: every student block's output, every frozen block's input (the drift-damping stream), ln_post's output
    def __init__(self, model: KICLIP):
        visual = model.model.visual
        self.cls: list[torch.Tensor] = []
        self.patch: list[torch.Tensor] = []
        self.stream: list[torch.Tensor] = []
        self.head: torch.Tensor | None = None
        for block in visual.transformer.resblocks:
            block.register_forward_hook(self._block)
        for block in visual.transformer.resblocks_dd or []:
            block.register_forward_pre_hook(self._stream)
        visual.ln_post.register_forward_hook(self._head)

    def _block(self, _module, _inputs, out: torch.Tensor) -> None:
        self.cls.append(out[:, 0].float())
        self.patch.append(out[:, 1:].float().mean(1))

    def _stream(self, _module, inputs: tuple[torch.Tensor, ...]) -> None:
        self.stream.append(inputs[0][:, 0].float())

    def _head(self, _module, _inputs, out: torch.Tensor) -> None:
        self.head = out

    def clear(self) -> None:
        self.cls, self.patch, self.stream, self.head = [], [], [], None


@dataclass
class Entry:
    name: str
    checkpoint: str
    model: KICLIP
    recorder: Recorder
    segments: int
    config: dict | None


def load_entry(spec: str, cfg: Config, device: torch.device, segments: int) -> Entry:
    if spec == CLIP_NAME:
        model = KICLIP(CLIP.from_state_dict(pretrained_state(cfg.model.arch)), cfg.data.num_frames)
        return Entry(CLIP_NAME, CLIP_NAME, model.to(device).eval(), Recorder(model), segments, None)
    from src.test import load_weights

    path = Path(spec) / "averaged.pt" if Path(spec).is_dir() else Path(spec)
    state, ckpt_cfg = load_weights(path)
    if ckpt_cfg is None:
        raise ValueError(f"{path} carries no config; dump.py needs an averaged.pt written by src/train.py")
    for key in ("num_frames", "sampling_rate"):
        if ckpt_cfg["data"][key] != getattr(cfg.data, key):
            raise ValueError(f"{path} was trained with data.{key}={ckpt_cfg['data'][key]}; pass --set data.{key}=...")
    known = {f.name for f in fields(ModelConfig)}
    model_cfg = ModelConfig(**{k: v for k, v in ckpt_cfg["model"].items() if k in known})
    weights = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    model = build_kiclip(replace(cfg, model=model_cfg), weights)
    model.load_state_dict(state, strict=False)
    own = model_cfg.cgas_segments if model.cgas_head is not None else segments
    name = path.parent.name
    return Entry(name, str(path), model.to(device).eval(), Recorder(model), own, ckpt_cfg)


@torch.no_grad()
def record(entry: Entry, video: torch.Tensor, amp: bool) -> dict[str, torch.Tensor]:
    # video (B, V, T, 3, H, W) -> per-frame tensors (B, V, T, ...) and per-video means over views and frames (B, ...)
    b, v, t = video.shape[:3]
    rec, visual = entry.recorder, entry.model.model.visual
    with torch.autocast(video.device.type, dtype=torch.float16, enabled=amp):
        _, _, _, _, cgas_raw = entry.model.encode_video(video)
        head = rec.head @ visual.proj
    cls = torch.stack(rec.cls, 1)                                              # (N, L, W)
    out = {
        "head": head.float().view(b, v, t, -1).half().cpu(),
        "pooled": pool_segments(cls, entry.segments, "mean").view(b, v, t, cls.shape[1], -1).half().cpu(),
        "cls": cls.view(b, v * t, *cls.shape[1:]).mean(1).cpu(),
        "patch": torch.stack(rec.patch, 1).view(b, v * t, *cls.shape[1:]).mean(1).cpu(),
    }
    if rec.stream:
        out["stream"] = torch.stack(rec.stream, 1).view(b, v * t, *cls.shape[1:]).mean(1).cpu()
    if cgas_raw is not None:
        out["cgas"] = cgas_raw.float().view(b, v, t, -1).half().cpu()
    rec.clear()
    return out


def collect(entries: list[Entry], batches: Iterable, device: torch.device, amp: bool) -> dict[str, dict]:
    chunks: dict[str, list[dict]] = {e.name: [] for e in entries}
    labels = []
    for video, y in batches:
        video = video.to(device, non_blocking=True)
        labels.append(y.cpu())
        for e in entries:
            chunks[e.name].append(record(e, video, amp))
    y = torch.cat(labels)
    return {name: {"labels": y, **{k: torch.cat([c[k] for c in parts]) for k in parts[0]}} for name, parts in chunks.items()}


@torch.no_grad()
def text_features(entries: list[Entry], tokens: torch.Tensor, device: torch.device, amp: bool) -> dict[str, torch.Tensor]:
    out = {}
    for e in entries:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out[e.name] = class_features(e.model.model, tokens.to(device)).float().cpu()  # (C, D), all prompts
    return out


def header(entry: Entry) -> dict:
    m = entry.model
    return {
        "name": entry.name,
        "checkpoint": entry.checkpoint,
        "config": entry.config,
        "logit_scale": m.model.logit_scale.exp().item(),
        "segments": entry.segments,
        "cgas_head": {k: v.float().cpu() for k, v in m.cgas_head.state_dict().items()} if m.cgas_head is not None else None,
        "splits": {},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="dump embeddings, CGAS inputs and block tokens for report.py")
    ap.add_argument("--runs", nargs="+", default=list(RUNS), help="run dirs or averaged.pt files")
    ap.add_argument("--no-clip", action="store_true", help="skip the zero-shot CLIP reference")
    ap.add_argument("--splits", nargs="+", default=None, help="default: data.train + data.test")
    ap.add_argument("--segments", type=int, default=64, help="pooling width for runs without a CGAS head")
    ap.add_argument("--config", action="append", default=[], help="dataset config (default configs/b2n/default/hmdb51.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL", help="override a config key")
    ap.add_argument("--out", default="runs/analysis/dumps")
    args = ap.parse_args()

    from src.data.dataset import VideoDataset
    from src.data.prompts import tokenize

    cfg = load_config(args.config or [REPO_ROOT / "configs" / "b2n" / "hmdb51.yaml"], overrides=parse_overrides(args.set))
    seed_everything(cfg.seed)
    device = pick_device()
    amp = cfg.test.amp and device.type == "cuda"
    specs = ([] if args.no_clip else [CLIP_NAME]) + args.runs
    entries = [load_entry(spec, cfg, device, args.segments) for spec in specs]
    names = [e.name for e in entries]
    if len(set(names)) != len(names):
        raise SystemExit(f"run names collide: {names}; dump them in separate calls with different --out")
    for e in entries:
        print(f"  {e.name:28s} {e.checkpoint} | cgas {'on' if e.model.cgas_head is not None else 'off'} "
              f"| drift-damping blocks {'on' if e.model.model.visual.transformer.resblocks_dd is not None else 'off'}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dumps = {e.name: header(e) for e in entries}
    for split in args.splits or [cfg.data.train, *cfg.data.test]:
        dataset = VideoDataset(cfg, split, train=False, views=cfg.test.views)
        loader = DataLoader(dataset, batch_size=cfg.test.batch_size, num_workers=cfg.data.num_workers,
                            pin_memory=device.type == "cuda")
        classes = text_features(entries, tokenize(dataset.classnames), device, amp)
        tensors = collect(entries, progress(loader, desc=split, ncols=100), device, amp)
        for e in entries:
            dumps[e.name]["splits"][split] = {"classnames": dataset.classnames, "classes": classes[e.name], **tensors[e.name]}
            torch.save(dumps[e.name], out / f"{e.name}.pt")
        print(f"  {split}: {len(dataset)} videos dumped for {len(entries)} models")
    print(f"  wrote {out}/<name>.pt; next: python scripts/analysis/report.py --dumps {out}")


if __name__ == "__main__":
    main()
