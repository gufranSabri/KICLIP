from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.models.clip import CLIP, pretrained_state, with_frozen_copy
from src.models.field import FieldEmbedder, build_field, sample_pool_mode

TEXT_CHUNK = 2048


def class_features(clip: CLIP, tokens: torch.Tensor) -> torch.Tensor:  # (P, C, L) -> (C, D), averaged over prompts
    p, c, l = tokens.shape
    text = torch.cat([clip.encode_text(chunk) for chunk in tokens.reshape(p * c, l).split(TEXT_CHUNK)])
    # reference/KICLIP normalizes each per-template encoding into a loop variable it discards, so the mean below is
    # over the UNnormalized per-template vectors; only this final normalize is real. Reproduced here as-is.
    return F.normalize(text.view(p, c, -1).mean(0), dim=-1)


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
    # reference/KICLIP's res_classifier, with a fresh MLP instead of a pretrained-ImageNet head
    return nn.Sequential(
        nn.Linear(dim, 768), nn.GELU(), nn.Linear(768, 768), nn.GELU(), nn.Linear(768, num_classes)
    )


class KICLIP(nn.Module):
    # the student CLIP, the frozen pretrained CLIP (teacher) it is distilled towards and injected from, the projectors

    def __init__(
        self,
        model: CLIP,
        num_frames: int,
        teacher: CLIP | None = None,
        inject: bool = False,
        distill: bool = False,
        alpha: float = 0.1,
        aux: bool = False,
        num_classes: int = 0,
        field_source: str | None = None,
        field_segments: int = 64,
        field_pool_modes: tuple[str, ...] = ("mean",),
        field_dropout_p: float = 0.3,
        field_alpha: float = 0.1,
        field_ce_weight: float = 1.0,
    ):
        super().__init__()
        needs_teacher = inject or distill or field_source in ("teacher", "both")
        if needs_teacher and teacher is None:
            raise ValueError("injecting from the teacher, distilling and field_source=teacher/both need a teacher")
        self.model, self.num_frames, self.inject, self.alpha = model, num_frames, inject, alpha
        self.teacher = teacher.eval().requires_grad_(False) if teacher is not None else None
        self.projector_v = residual_projector(model.embed_dim) if distill else None
        self.projector_t = residual_projector(model.embed_dim) if distill else None
        # reference/KICLIP always builds this alongside distillation and trains it on the pooled, projected embedding
        self.aux = aux_classifier(model.embed_dim, num_classes) if aux else None
        self.field_source = field_source
        self.field_segments, self.field_pool_modes = field_segments, list(field_pool_modes)
        self.field_dropout_p, self.field_alpha, self.field_ce_weight = field_dropout_p, field_alpha, field_ce_weight
        field_channels = 2 if field_source == "both" else 1
        # field_embed's output serves both as its own classifier (field_ce) and, scaled by field_alpha, as a
        # residual correction on the FINAL embedding (forward) -- not mid-stack, where downstream blocks' own
        # pretrained biases silently reconstruct a default and erase the injection's effect
        self.field_embed = (
            FieldEmbedder(
                len(model.visual.transformer.resblocks), field_segments, model.embed_dim, field_channels,
            ) if field_source is not None else None
        )

    # the teacher is the pretrained CLIP, rebuilt whenever a model is built, so checkpoints leave it out
    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        return {k: v for k, v in super().state_dict(*args, **kwargs).items() if not k.startswith("teacher.")}

    def load_state_dict(self, state, strict: bool = True, assign: bool = False):
        teacher = {k: v for k, v in super().state_dict().items() if k.startswith("teacher.")}
        return super().load_state_dict({**teacher, **state}, strict, assign)

    def encode_video(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor], list[torch.Tensor], torch.Tensor | None]:
        # (..., T, 3, H, W) -> the student's and the teacher's frame features (frames, D), unit norm (already
        # field-corrected when field_source is set), every vision block's own output (only when field_source is
        # set), and the RAW (unnormalized) field correction, for field_ce's own classification loss
        needs_teacher_hidden = self.field_source in ("teacher", "both")
        x = video.flatten(0, -4)
        stream = target = None
        student_hidden: list[torch.Tensor] = []
        teacher_hidden: list[torch.Tensor] = []
        if self.teacher is not None:
            with torch.no_grad():
                if needs_teacher_hidden:
                    # teacher_hidden: every block's own output, for the field; stream: every block's input (the
                    # embedding, then every output but the last), for injection -- same as trace()'s "inputs"
                    embed = self.teacher.visual.embed(x)
                    teacher_hidden = self.teacher.visual.transformer.hidden(embed)
                    stream = [embed] + teacher_hidden[:-1]
                    target = self.teacher.visual.head(teacher_hidden[-1])
                else:
                    target, stream = self.teacher.visual.trace(x)
            target = F.normalize(target, dim=-1)
        if self.field_source is not None:
            student_hidden = self.model.visual.hidden(x, stream if self.inject else None)
            frames = self.model.visual.head(student_hidden[-1])
        else:
            frames = self.model.visual(x, stream if self.inject else None)
        field_raw = None
        if self.field_source is not None:
            if self.training:
                field_raw = self._field_embed(student_hidden, teacher_hidden, sample_pool_mode(self.field_pool_modes))
            else:
                # training resamples a pool mode every step; eval instead averages field_embed's OUTPUT over
                # every non-dropout mode actually used in training, so eval sees the same mixture of modes
                # field_embed was trained on, rather than only ever exercising a single ('mean') branch of it
                modes = [m for m in self.field_pool_modes if m != "dropout"] or ["mean"]
                field_raw = torch.stack(
                    [self._field_embed(student_hidden, teacher_hidden, m) for m in modes]
                ).mean(0)
            frames = frames + self.field_alpha * field_raw
        return F.normalize(frames, dim=-1), target, student_hidden, teacher_hidden, field_raw

    def _field_embed(
        self, student_hidden: list[torch.Tensor], teacher_hidden: list[torch.Tensor], mode: str
    ) -> torch.Tensor:
        channels = []
        if self.field_source in ("student", "both"):
            channels.append(torch.stack(
                [build_field(h, self.field_segments, mode, self.field_dropout_p) for h in student_hidden], dim=1
            ))
        if self.field_source in ("teacher", "both"):
            channels.append(torch.stack(
                [build_field(h, self.field_segments, mode, self.field_dropout_p) for h in teacher_hidden], dim=1
            ))
        stacked = channels[0] if len(channels) == 1 else torch.cat(channels, dim=-1)  # (N, L, C*M)
        return self.field_embed(stacked)  # (N, D), unnormalized

    def forward(
        self, video: torch.Tensor, tokens: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # -> logits (B, C), the distillation loss, the aux classifier's CE, the field embedder's CE
        frames, teacher_frames, student_hidden, teacher_hidden, field_raw = self.encode_video(video)  # (B*T, D)
        classes = class_features(self.model, tokens)                                       # (C, D)
        logits = clip_logits(self.model, frames, classes, self.num_frames)  # frames already carries field_alpha * field_raw
        zero = torch.zeros((), device=logits.device)
        pooled_frames = frames
        kd = zero
        if self.projector_v is not None:
            with torch.no_grad():
                teacher_classes = class_features(self.teacher, tokens)
            pooled_frames = frames + self.alpha * self.projector_v(frames)
            kd = distillation_loss(
                pooled_frames,
                classes + self.alpha * self.projector_t(classes),
                teacher_frames,
                teacher_classes,
            )
        aux = zero
        if self.aux is not None and labels is not None:
            pooled = pooled_frames.view(-1, self.num_frames, pooled_frames.shape[-1]).mean(1)  # (B, D)
            aux = F.cross_entropy(self.aux(pooled), labels)
        field_ce = zero
        if self.field_embed is not None and labels is not None:
            # F.normalize runs in fp32 under autocast regardless of field_raw's dtype; match classes' dtype
            # before the matmul in clip_logits, which autocast does NOT upcast for us here
            field_normed = F.normalize(field_raw, dim=-1).to(classes.dtype)
            field_logits = clip_logits(self.model, field_normed, classes, self.num_frames)
            field_ce = F.cross_entropy(field_logits, labels)
        return logits, kd, aux, field_ce


def build_kiclip(cfg: Config, weights: dict[str, torch.Tensor] | None = None, num_classes: int = 0) -> KICLIP:
    # weights: a trained student to evaluate; None builds a student to train, starting from the pretrained CLIP
    m = cfg.model
    blocks = m.inject and m.inject_source == "blocks"
    from_teacher = m.inject and m.inject_source == "teacher"
    distill = weights is None and m.distill_weight > 0
    aux = weights is None and m.aux_classifier
    field_needs_teacher = m.field_enabled and m.field_source in ("teacher", "both")
    pretrained = pretrained_state(m.arch) if weights is None or from_teacher or field_needs_teacher else None
    if weights is None:
        weights = with_frozen_copy(pretrained) if blocks else pretrained
    student = CLIP.from_state_dict(
        weights,
        temporal=m.temporal,
        num_frames=cfg.data.num_frames,
        inject=blocks,
        grad_checkpointing=m.grad_checkpointing,
    )
    teacher = (
        CLIP.from_state_dict(pretrained) if distill or from_teacher or field_needs_teacher
        else None
    )
    return KICLIP(
        student, cfg.data.num_frames, teacher, inject=from_teacher, distill=distill, alpha=m.residual_alpha,
        aux=aux, num_classes=num_classes,
        field_source=m.field_source if m.field_enabled else None,
        field_segments=m.field_segments,
        field_pool_modes=tuple(m.field_pool_modes),
        field_dropout_p=m.field_dropout_p,
        field_alpha=m.field_alpha,
        field_ce_weight=m.field_ce_weight,
    )
