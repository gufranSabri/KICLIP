from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.models.clip import CLIP, with_frozen_copy

TEXT_CHUNK = 2048


def frame_features(clip: CLIP, video: torch.Tensor) -> torch.Tensor:  # (..., T, 3, H, W) -> (frames, D), unit norm
    return F.normalize(clip.encode_image(video.flatten(0, -4)), dim=-1)


def class_features(clip: CLIP, tokens: torch.Tensor) -> torch.Tensor:  # (P, C, L) -> (C, D), averaged over prompts
    p, c, l = tokens.shape
    text = torch.cat([clip.encode_text(chunk) for chunk in tokens.reshape(p * c, l).split(TEXT_CHUNK)])
    return F.normalize(F.normalize(text, dim=-1).view(p, c, -1).mean(0), dim=-1)


def clip_logits(clip: CLIP, frames: torch.Tensor, classes: torch.Tensor, num_frames: int) -> torch.Tensor:
    logits = clip.logit_scale.exp() * frames @ classes.T  # (clips * T, C)
    return logits.view(-1, num_frames, logits.shape[-1]).mean(1)


def distillation_loss(
    frames: torch.Tensor, classes: torch.Tensor, teacher_frames: torch.Tensor, teacher_classes: torch.Tensor
) -> torch.Tensor:
    return (
        2
        - F.cosine_similarity(frames, teacher_frames, dim=-1).mean()
        - F.cosine_similarity(classes, teacher_classes, dim=-1).mean()
    )


def residual_projector(dim: int) -> nn.Sequential:
    proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.GELU(), nn.Linear(dim, dim, bias=False))
    nn.init.kaiming_normal_(proj[0].weight)
    nn.init.zeros_(proj[2].weight)
    return proj


def aux_classifier(dim: int, num_classes: int) -> nn.Sequential:
    from torchvision.models import ViT_B_16_Weights

    # the middle layer is torchvision's ImageNet ViT-B/16 head, pretrained
    state = ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
    head = nn.Linear(768, 1000)
    head.load_state_dict({"weight": state["heads.head.weight"], "bias": state["heads.head.bias"]})
    return nn.Sequential(nn.Linear(dim, 768), nn.GELU(), head, nn.GELU(), nn.Linear(1000, num_classes))


class KICLIP(nn.Module):
    # the student CLIP, the projectors distilled towards the frozen teacher, and the closed-set aux classifier

    def __init__(
        self,
        model: CLIP,
        num_frames: int,
        distill: bool = True,
        alpha: float = 0.1,
        aux_classes: int | None = None,
    ):
        super().__init__()
        self.model, self.num_frames, self.alpha = model, num_frames, alpha
        self.projector_v = residual_projector(model.embed_dim) if distill else None
        self.projector_t = residual_projector(model.embed_dim) if distill else None
        self.aux_classifier = aux_classifier(model.embed_dim, aux_classes) if aux_classes else None

    def forward(self, video: torch.Tensor, tokens: torch.Tensor):
        frames = frame_features(self.model, video)    # (B*T, D)
        classes = class_features(self.model, tokens)  # (C, D)
        logits = clip_logits(self.model, frames, classes, self.num_frames)
        if self.projector_v is not None:
            frames = frames + self.alpha * self.projector_v(frames)
            classes = classes + self.alpha * self.projector_t(classes)
        aux = None
        if self.aux_classifier is not None:
            aux = self.aux_classifier(frames.view(-1, self.num_frames, frames.shape[-1]).mean(1))  # (B, C_train)
        if self.projector_v is None:
            return logits, None, None, aux
        return logits, frames, classes, aux


def build_kiclip(cfg: Config, state: dict[str, torch.Tensor], num_classes: int) -> KICLIP:
    m = cfg.model
    clip = CLIP.from_state_dict(
        with_frozen_copy(state) if m.inject else state,
        temporal=m.temporal,
        num_frames=cfg.data.num_frames,
        inject=m.inject,
        grad_checkpointing=m.grad_checkpointing,
    )
    return KICLIP(
        clip,
        cfg.data.num_frames,
        distill=m.distill_weight > 0,
        alpha=m.residual_alpha,
        aux_classes=num_classes if m.aux_weight > 0 else None,
    )
