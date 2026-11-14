from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_MOTION_SOURCES = ("rgb_diff", "embed_diff")


def _consecutive_diff(x: torch.Tensor, num_frames: int, feature_dims: int) -> torch.Tensor:
    # (N, *feature) -> (N, *feature), x[t+1] - x[t] within each num_frames-sized clip, last frame zeroed
    shape = x.shape
    feature = shape[-feature_dims:] if feature_dims else ()
    x = x.reshape(-1, num_frames, *feature)
    diff = x[:, 1:] - x[:, :-1]
    zero = x.new_zeros(x.shape[0], 1, *feature)
    return torch.cat([diff, zero], 1).reshape(shape)


def rgb_diff(frames: torch.Tensor, num_frames: int, patch_size: int) -> torch.Tensor:
    # (N, 3, H, W), N = ... * T -> (N, P, 3), average-pooled to the patch grid
    diff = _consecutive_diff(frames, num_frames, feature_dims=3)
    pooled = F.avg_pool2d(diff, patch_size)  # (N, 3, H/patch, W/patch)
    return pooled.flatten(2).transpose(1, 2)


def embed_diff(frames: torch.Tensor, num_frames: int, conv1: nn.Conv2d) -> torch.Tensor:
    # (N, 3, H, W) -> (N, P, W_embed), diffed in CLIP's own patch-embedding space
    embedded = conv1(frames).flatten(2).transpose(1, 2)  # (N, P, W_embed)
    return _consecutive_diff(embedded, num_frames, feature_dims=2)


def motion_dim(source: str, width: int) -> int:
    return 3 if source == "rgb_diff" else width


class MotionEmbedder(nn.Module):
    def __init__(self, motion_dim: int, width: int, bottleneck: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(motion_dim, bottleneck), nn.GELU(), nn.Linear(bottleneck, width, bias=False)
        )
        nn.init.zeros_(self.proj[-1].weight)  # starts as a no-op, like kiclip.residual_projector

    def forward(self, motion: torch.Tensor) -> torch.Tensor:  # (N, P, motion_dim) -> (N, P, width)
        return self.proj(motion)
