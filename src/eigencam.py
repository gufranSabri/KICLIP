# EigenCAM over the vision blocks, computed as pytorch-grad-cam does (2-D projection, ReLU, min-max, bilinear, jet overlay).

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config
from src.data.video import CLIP_MEAN, CLIP_STD, eval_transform, read_clips
from src.models.clip import CLIP, has_injection, pretrained_state
from src.models.kiclip import KICLIP, build_kiclip


def load_vision_model(spec: str, cfg: Config) -> KICLIP:
    # spec: "clip" for the pretrained (teacher) CLIP, else averaged.pt or an original FROSTER / KICLIP .pth
    if spec == "clip":
        return KICLIP(CLIP.from_state_dict(pretrained_state(cfg.model.arch)), cfg.data.num_frames).eval()
    from src.test import injection, load_weights

    weights, ckpt_cfg = load_weights(Path(spec))
    if ckpt_cfg is None:  # the original releases inject from frozen block copies, if at all
        temporal, num_frames = cfg.model.temporal, cfg.data.num_frames
        source = "blocks" if has_injection(weights) else None
    else:
        temporal, num_frames = ckpt_cfg["model"]["temporal"], ckpt_cfg["data"]["num_frames"]
        source = injection(ckpt_cfg["model"])
    if num_frames != cfg.data.num_frames:
        raise ValueError(f"{spec} was trained on {num_frames}-frame clips; pass --set data.num_frames={num_frames}")
    model = replace(cfg.model, temporal=temporal, inject=source is not None, inject_source=source or "teacher")
    return build_kiclip(replace(cfg, model=model), weights).eval()


def load_clip(path: Path, cfg: Config, view: int) -> tuple[torch.Tensor, np.ndarray]:
    # one test view: (T, 3, S, S) model input and (T, S, S, 3) RGB in [0, 1] for display
    d = cfg.data
    clips = read_clips(path, d.num_frames, d.num_frames * d.sampling_rate / d.target_fps, cfg.test.views)
    x = eval_transform(clips, d.train_scales[0], d.crop_size)[view]
    rgb = (x * CLIP_STD + CLIP_MEAN).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    return x, rgb


@torch.no_grad()
def layer_tokens(model: KICLIP, frames: torch.Tensor) -> list[torch.Tensor]:  # (T, 3, R, R) -> per block (T, 1 + P, W)
    outs: list[torch.Tensor] = []
    hooks = [
        block.register_forward_hook(lambda _m, _i, out: outs.append(out.float().cpu()))
        for block in model.model.visual.transformer.resblocks
    ]
    try:
        model.encode_video(frames)
    finally:
        for h in hooks:
            h.remove()
    return outs


def eigencam(tokens: torch.Tensor, size: int) -> np.ndarray:  # (T, 1 + P, W) -> (T, size, size) in [0, 1]
    patches = tokens[:, 1:].numpy()
    grid = int(round(patches.shape[1] ** 0.5))
    cams = []
    for a in patches:
        a = a - a.mean(0)
        _, _, vt = np.linalg.svd(a, full_matrices=False)
        cams.append(np.maximum(a @ vt[0], 0).reshape(grid, grid))
    cam = torch.from_numpy(_minmax(np.stack(cams)))[:, None]
    cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)[:, 0].numpy()
    return _minmax(cam)


def overlay(rgb: np.ndarray, cam: np.ndarray, image_weight: float = 0.5) -> np.ndarray:
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    out = (1 - image_weight) * heat + image_weight * rgb
    return out / out.max()


def _minmax(x: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, x.ndim))
    x = x - x.min(axis=axes, keepdims=True)
    return x / (x.max(axis=axes, keepdims=True) + 1e-7)
