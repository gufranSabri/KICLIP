from __future__ import annotations

import math
import random
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def read_clips(path: Path, num_frames: int, span: float, views: int | None) -> torch.Tensor:
    # views=None: one random clip; otherwise `views` clips whose starts are spread evenly over the video
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # DataLoader workers already decode in parallel; FFmpeg threads on top exhaust the task limit
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate or stream.guessed_rate or 30)
        offset = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        delta = max(_duration(container, stream, fps) - span, 0.0)
        starts = [random.uniform(0.0, delta)] if views is None else [delta * v / views for v in range(views)]
        steps = np.linspace(0.0, max(span - 1.0 / fps, 0.0), num_frames)
        times = (np.asarray(starts)[:, None] + steps[None, :]).ravel() + offset
        frames = _pick(container, stream, times)
    out = torch.from_numpy(np.stack(frames))  # (V*T, H, W, 3)
    return out.view(len(starts), num_frames, *out.shape[1:])


def _duration(container, stream, fps: float) -> float:
    if stream.duration:
        return float(stream.duration * stream.time_base)
    if container.duration:
        return container.duration / av.time_base
    if stream.frames:
        return stream.frames / fps
    raise ValueError("no duration in the container metadata")


def _pick(container, stream, times: np.ndarray) -> list[np.ndarray]:
    # for every target time, the last frame shown at or before it
    order = np.argsort(times, kind="stable")
    if times[order[0]] > 0:
        container.seek(int(times[order[0]] / stream.time_base), stream=stream)
    picked: list = [None] * len(times)
    j, prev = 0, None
    for frame in container.decode(stream):
        if frame.time is None:
            continue
        while j < len(order) and times[order[j]] < frame.time:
            picked[order[j]] = frame if prev is None else prev
            j += 1
        if j == len(order):
            break
        prev = frame
    if j < len(order):
        if prev is None:
            raise ValueError("no decodable frames")
        for k in order[j:]:
            picked[k] = prev
    arrays: dict[int, np.ndarray] = {}
    for f in picked:
        if id(f) not in arrays:
            arrays[id(f)] = f.to_ndarray(format="rgb24")
    return [arrays[id(f)] for f in picked]


def _to_float(frames: torch.Tensor) -> torch.Tensor:
    return frames.permute(0, 3, 1, 2).float().div_(255)  # (N, H, W, 3) uint8 -> (N, 3, H, W)


def _short_side(x: torch.Tensor, size: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    if min(h, w) == size:
        return x
    if w < h:
        new_h, new_w = int(math.floor(h / w * size)), size
    else:
        new_h, new_w = size, int(math.floor(w / h * size))
    return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - CLIP_MEAN) / CLIP_STD


def train_transform(clip: torch.Tensor, scales: tuple[int, int], crop: int, flip: bool) -> torch.Tensor:
    x = _short_side(_to_float(clip), round(random.uniform(*scales)))  # (T, 3, H, W)
    i = random.randint(0, x.shape[-2] - crop)
    j = random.randint(0, x.shape[-1] - crop)
    x = x[..., i:i + crop, j:j + crop]
    if flip and random.random() < 0.5:
        x = x.flip(-1)
    return _normalize(x)


def eval_transform(clips: torch.Tensor, size: int, crop: int) -> torch.Tensor:
    v, t = clips.shape[:2]
    x = _short_side(_to_float(clips.flatten(0, 1)), size)
    i = math.ceil((x.shape[-2] - crop) / 2)
    j = math.ceil((x.shape[-1] - crop) / 2)
    return _normalize(x[..., i:i + crop, j:j + crop]).view(v, t, 3, crop, crop)
