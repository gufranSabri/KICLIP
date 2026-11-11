# OpenAI CLIP as plain modules; parameter names match the OpenAI / open_clip checkpoints so weights load strictly.

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.config import VALID_ARCHS


def pretrained_state(arch: str) -> dict[str, torch.Tensor]:
    import open_clip
    from open_clip.factory import load_state_dict

    if arch not in VALID_ARCHS:
        raise ValueError(f"model.arch must be one of {VALID_ARCHS}, got {arch!r}")
    return load_state_dict(open_clip.download_pretrained(open_clip.get_pretrained_cfg(arch, "openai")))


INJECTED = "visual.transformer.resblocks_f."


def with_frozen_copy(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # seeds the knowledge-injection blocks with the pretrained vision blocks
    blocks = "visual.transformer.resblocks."
    return {**state, **{INJECTED + k[len(blocks):]: v for k, v in state.items() if k.startswith(blocks)}}


def has_injection(state: dict[str, torch.Tensor]) -> bool:
    return any(k.startswith(INJECTED) for k in state)


def expand_temporal(x: torch.Tensor, num_frames: int) -> torch.Tensor:
    # each frame's tokens plus its next and previous frame's; zero tokens at clip ends still take softmax mass, as in FROSTER
    bt, h, n, d = x.shape
    if bt % num_frames:
        raise ValueError(f"{bt} frames do not split into {num_frames}-frame clips")
    x = x.reshape(bt // num_frames, num_frames, h, n, d)
    zero = x.new_zeros(x.shape[0], 1, h, n, d)
    nxt = torch.cat([x[:, 1:], zero], 1)
    prev = torch.cat([zero, x[:, :-1]], 1)
    return torch.cat([nxt, prev, x], 3).reshape(bt, h, 3 * n, d)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).to(x.dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, causal: bool = False, temporal: str | None = None, num_frames: int = 1):
        super().__init__()
        self.heads, self.causal, self.temporal, self.num_frames = heads, causal, temporal, num_frames
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.in_proj_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, l, d = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.view(n, l, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)  # each (N, H, L, d)
        if self.temporal == "expand":
            k, v = expand_temporal(k, self.num_frames), expand_temporal(v, self.num_frames)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.out_proj(out.transpose(1, 2).reshape(n, l, d))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, causal: bool = False, temporal: str | None = None, num_frames: int = 1):
        super().__init__()
        self.ln_1 = LayerNorm(dim)
        self.attn = Attention(dim, heads, causal, temporal, num_frames)
        self.ln_2 = LayerNorm(dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(dim * 4, dim)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        causal: bool = False,
        temporal: str | None = None,
        num_frames: int = 1,
        inject: bool = False,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.grad_checkpointing = grad_checkpointing
        self.resblocks = nn.ModuleList(
            ResidualAttentionBlock(width, heads, causal, temporal, num_frames) for _ in range(layers)
        )
        # knowledge injection: a frozen frame-wise copy of the pretrained blocks, fed the same tokens
        self.resblocks_f = (
            nn.ModuleList(ResidualAttentionBlock(width, heads, causal) for _ in range(layers)).requires_grad_(False)
            if inject else None
        )

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor, frozen: list[torch.Tensor] | None = None) -> torch.Tensor:  # (N, L, D)
        # frozen: another model's input to every block (see trace), averaged in before the block like resblocks_f's
        t = x
        for i, block in enumerate(self.resblocks):
            if self.resblocks_f is not None:
                x, t = (x + t) / 2, self._run(self.resblocks_f[i], t)
            elif frozen is not None:
                x = (x + frozen[i]) / 2
            x = self._run(block, x)
        return x

    def trace(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:  # output and every block's input
        inputs = []
        for block in self.resblocks:
            inputs.append(x)
            x = self._run(block, x)
        return x, inputs

    def hidden(self, x: torch.Tensor, frozen: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
        # every block's own output, (N, L, D) each; frozen: as in forward, injection's effect on the input
        t, outputs = x, []
        for i, block in enumerate(self.resblocks):
            if self.resblocks_f is not None:
                x, t = (x + t) / 2, self._run(self.resblocks_f[i], t)
            elif frozen is not None:
                x = (x + frozen[i]) / 2
            x = self._run(block, x)
            outputs.append(x)
        return outputs


class VisionTransformer(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        temporal: str | None = None,
        num_frames: int = 1,
        inject: bool = False,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        scale = width ** -0.5
        self.conv1 = nn.Conv2d(3, width, patch_size, patch_size, bias=False)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(
            width, layers, heads, temporal=temporal, num_frames=num_frames, inject=inject,
            grad_checkpointing=grad_checkpointing,
        )
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def embed(self, x: torch.Tensor) -> torch.Tensor:  # (N, 3, R, R) -> (N, 1 + P, W), the first block's input
        if x.shape[-2:] != (self.resolution, self.resolution):
            raise ValueError(
                f"expected {self.resolution}px frames, got {tuple(x.shape[-2:])}; set data.crop_size={self.resolution}"
            )
        x = self.conv1(x).flatten(2).transpose(1, 2)                       # (N, P, W)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], 1) + self.positional_embedding.to(x.dtype)  # (N, 1 + P, W)
        return self.ln_pre(x)

    def head(self, x: torch.Tensor) -> torch.Tensor:  # (N, 1 + P, W) -> (N, D)
        return self.ln_post(x[:, 0]) @ self.proj

    def forward(self, x: torch.Tensor, frozen: list[torch.Tensor] | None = None) -> torch.Tensor:  # -> (N, D)
        return self.head(self.transformer(self.embed(x), frozen))

    def trace(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:  # (N, D) and every block's input
        x, inputs = self.transformer.trace(self.embed(x))
        return self.head(x), inputs

    def hidden(self, x: torch.Tensor, frozen: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
        # every block's own output, (N, 1 + P, W) each
        return self.transformer.hidden(self.embed(x), frozen)


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        resolution: int,
        vision_layers: int,
        vision_width: int,
        patch_size: int,
        context_length: int,
        vocab_size: int,
        text_width: int,
        text_heads: int,
        text_layers: int,
        temporal: str | None = None,
        num_frames: int = 1,
        inject: bool = False,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.visual = VisionTransformer(
            resolution, patch_size, vision_width, vision_layers, vision_width // 64, embed_dim,
            temporal=temporal, num_frames=num_frames, inject=inject, grad_checkpointing=grad_checkpointing,
        )
        self.transformer = Transformer(
            text_width, text_layers, text_heads, causal=True, grad_checkpointing=grad_checkpointing
        )
        self.token_embedding = nn.Embedding(vocab_size, text_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, text_width).normal_(std=0.01))
        self.ln_final = LayerNorm(text_width)
        self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim).normal_(std=text_width ** -0.5))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor], **kwargs) -> CLIP:
        model = cls(**_dims(state), **kwargs)
        model.load_state_dict(state)
        return model

    @property
    def embed_dim(self) -> int:
        return self.text_projection.shape[1]

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:  # (N, L) -> (N, D)
        x = self.token_embedding(tokens) + self.positional_embedding
        x = self.ln_final(self.transformer(x))
        # read out at end-of-text, the highest token id in each sequence
        return x[torch.arange(x.shape[0], device=x.device), tokens.argmax(-1)] @ self.text_projection


def _dims(state: dict[str, torch.Tensor]) -> dict[str, int]:
    def layers(prefix: str) -> int:
        return sum(1 for k in state if k.startswith(prefix) and k.endswith(".attn.in_proj_weight"))

    patch = state["visual.conv1.weight"].shape[-1]
    grid = round((state["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    text_width = state["ln_final.weight"].shape[0]
    return {
        "embed_dim": state["text_projection"].shape[1],
        "resolution": patch * grid,
        "vision_layers": layers("visual.transformer.resblocks."),
        "vision_width": state["visual.conv1.weight"].shape[0],
        "patch_size": patch,
        "context_length": state["positional_embedding"].shape[0],
        "vocab_size": state["token_embedding.weight"].shape[0],
        "text_width": text_width,
        "text_heads": text_width // 64,
        "text_layers": layers("transformer.resblocks."),
    }
