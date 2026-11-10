#!/usr/bin/env python3
# EigenCAM grid: one row per model, columns = RGB frame + chosen vision blocks; writes PNG + PDF.
#
#     python scripts/figures/eigencam.py --split b2n/hmdb51/novel --index 0 \
#         --checkpoints runs/b2n_hmdb51_s1_froster/averaged.pt runs/b2n_hmdb51_s1_kiclip/averaged.pt \
#         --labels FROSTER KICLIP
#     python scripts/figures/eigencam.py --video clip.mp4 --checkpoints clip runs/cross_k400_kiclip/averaged.pt

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.config import load_config, parse_overrides  # noqa: E402
from src.data.dataset import VideoDataset  # noqa: E402
from src.eigencam import eigencam, layer_tokens, load_clip, load_vision_model, overlay  # noqa: E402
from src.utils.seed import pick_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="EigenCAM of every model's vision blocks on one test clip")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="path to a video file")
    source.add_argument("--split", help="split to take the video from, e.g. b2n/hmdb51/novel (with --index)")
    ap.add_argument("--index", type=int, default=0, help="row of --split")
    ap.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="averaged.pt files, original FROSTER / KICLIP .pth files, or 'clip' for the pretrained CLIP",
    )
    ap.add_argument("--labels", nargs="+", default=None, help="row labels (default: run folder names)")
    ap.add_argument("--layers", nargs="+", type=int, default=list(range(5, 13)), help="1-indexed vision blocks")
    ap.add_argument("--frames", nargs="+", type=int, default=None, help="frames of the clip (default: the middle)")
    ap.add_argument("--view", type=int, default=None, help="which of the test.views clips (default: the middle)")
    ap.add_argument("--config", action="append", default=[], help="config(s) for data_root and the clip settings")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL", help="override a config key")
    ap.add_argument("--out", default=None, help="output .png (default: docs/figures/eigencam/<video>.png)")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    labels = args.labels or ["CLIP" if c == "clip" else Path(c).parent.name for c in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise SystemExit(f"--labels has {len(labels)} entries for {len(args.checkpoints)} checkpoints")
    frames = args.frames if args.frames is not None else [cfg.data.num_frames // 2]
    view = args.view if args.view is not None else cfg.test.views // 2
    if not 0 <= view < cfg.test.views or any(not 0 <= f < cfg.data.num_frames for f in frames):
        raise SystemExit(f"--view must be in [0, {cfg.test.views}) and --frames in [0, {cfg.data.num_frames})")

    title = ""
    if args.split:
        dataset = VideoDataset(cfg, args.split, train=False)
        if not 0 <= args.index < len(dataset):
            raise SystemExit(f"--index must be in [0, {len(dataset)}) for {args.split}")
        path = dataset.paths[args.index]
        title = dataset.classnames[dataset.labels[args.index]].split(":")[0]
    else:
        path = Path(args.video)
        if not path.exists():
            raise SystemExit(f"no video at {path}")

    device = pick_device()
    clip, rgb = load_clip(path, cfg, view)
    rows = []
    for spec in args.checkpoints:
        model = load_vision_model(spec, cfg).to(device)
        depth = len(model.model.visual.transformer.resblocks)
        if any(not 1 <= layer <= depth for layer in args.layers):
            raise SystemExit(f"--layers must be in [1, {depth}] for {spec}")
        tokens = layer_tokens(model, clip.to(device))
        rows.append([eigencam(tokens[layer - 1], cfg.data.crop_size) for layer in args.layers])
        blocks = model.model.visual.transformer.resblocks_f is not None
        source = "blocks" if blocks else "teacher" if model.inject else "nothing"
        print(f"  {spec}: {depth} blocks, injection from {source}")
        del model
        torch.cuda.empty_cache()

    out = Path(args.out) if args.out else REPO_ROOT / "docs" / "figures" / "eigencam" / f"{path.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["RGB IMAGE"] + [f"LAYER {layer}" for layer in args.layers]
    for f in frames:
        fig, axes = plt.subplots(len(rows), len(columns), figsize=(2 * len(columns), 2 * len(rows) + 0.6),
                                 squeeze=False)
        for r, (label, cams) in enumerate(zip(labels, rows)):
            images = [rgb[f]] + [overlay(rgb[f], cam[f]) for cam in cams]
            for c, image in enumerate(images):
                ax = axes[r, c]
                ax.imshow(image)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if c == 0:
                    ax.set_ylabel(label, fontsize=11, fontweight="bold")
                if r == len(rows) - 1:
                    ax.set_xlabel(columns[c], fontsize=11, fontweight="bold")
        fig.suptitle(f"{path.name}{' | ' + title if title else ''} | view {view}, frame {f}", fontsize=11)
        fig.tight_layout()
        dest = out if len(frames) == 1 else out.with_name(f"{out.stem}_f{f}{out.suffix}")
        for ext in (".png", ".pdf"):
            fig.savefig(dest.with_suffix(ext), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {dest.with_suffix('.png')} (+ .pdf)")


if __name__ == "__main__":
    main()
