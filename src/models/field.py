from __future__ import annotations

import random

import torch
import torch.nn as nn

POOL_MODES = ("mean", "max", "min", "strided", "dropout")


def pool_segments(raw: torch.Tensor, n_segments: int, mode: str, dropout_p: float = 0.3) -> torch.Tensor:
    # raw: (..., D) -> (..., n_segments); D must be divisible by n_segments, ported from QKV-Lens tensor_ops.py
    if mode not in POOL_MODES:
        raise ValueError(f"pool mode must be one of {POOL_MODES}, got {mode!r}")

    d = raw.shape[-1]
    if n_segments <= 0:
        raise ValueError(f"n_segments must be positive, got {n_segments}")
    if d % n_segments != 0:
        raise ValueError(f"feature dim D={d} is not divisible by n_segments={n_segments}")

    segment_size = d // n_segments
    if mode == "strided":
        return raw[..., ::segment_size][..., :n_segments]

    chunked = raw.reshape(*raw.shape[:-1], n_segments, segment_size)
    if mode == "max":
        return chunked.amax(dim=-1)
    if mode == "min":
        return chunked.amin(dim=-1)
    if mode == "dropout":
        return _dropout_pool(chunked, dropout_p)
    return chunked.mean(dim=-1)


def _dropout_pool(chunked: torch.Tensor, p: float) -> torch.Tensor:
    # zero a random p fraction of each chunk's elements, mean over survivors, rescaled 1/(1-p)
    keep = torch.rand_like(chunked) >= p
    survivors = keep.sum(dim=-1).clamp(min=1)  # guard: never divide by zero when a chunk drops everything
    return (chunked * keep).sum(dim=-1) / survivors


def sample_pool_mode(modes: list[str]) -> str:
    return random.choice(modes)


def build_field(hidden: torch.Tensor, n_segments: int, mode: str, dropout_p: float = 0.3) -> torch.Tensor:
    # hidden: (N, 1+P, W) -> (N, M): CLS token only (the token the final embedding is itself built from,
    # head(x) = ln_post(x[:, 0]) @ proj), feature dim pooled to n_segments
    return pool_segments(hidden[:, 0], n_segments, mode, dropout_p)


class FieldEmbedder(nn.Module):
    # open-vocabulary, unlike a closed-set linear head: maps the whole stacked field to the CLIP text tower's
    # embed_dim, so it can be compared by cosine similarity against ANY split's class text embeddings (base,
    # novel, or otherwise), the same way the main video embedding already is via clip_logits.
    # the output also gets added into the final pooled embedding (kiclip.py), a residual correction, not just a
    # standalone classifier head -- injecting mid-stack (every block) let the DOWNSTREAM blocks' own pretrained
    # Linear biases reconstruct a full-magnitude, image-independent default whenever the stream was driven far
    # from its original content, masking any effect; operating once on the finished embedding removes that
    # escape hatch. only the last layer's BIAS is zero-init, not its weight: field_raw exactly 0 made
    # F.normalize(field_raw) (field_ce, below) backward to nan under fp16 AMP, which silently skipped the
    # optimizer step for the WHOLE model every batch; keeping the weight's small random init means field_raw is
    # never exactly 0, so this is a near-no-op at init rather than an exact one
    def __init__(self, n_layers: int, n_segments: int, embed_dim: int, n_channels: int = 1):
        super().__init__()
        in_dim = n_channels * n_layers * n_segments
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(in_dim, 768), nn.GELU(), nn.Linear(768, embed_dim)
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fields: torch.Tensor) -> torch.Tensor:  # (N, L, channels*M) -> (N, embed_dim)
        return self.net(fields)
