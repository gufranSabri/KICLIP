from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.models.clip import CLIP, _alias_legacy_damping_keys, pretrained_state, with_frozen_copy
from src.models.cgas import CGASHead, build_cgas_input, sample_pool_mode

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


def cgas_loss(
    logits: torch.Tensor, labels: torch.Tensor, target: str, tau: float, teacher_logits: torch.Tensor | None
) -> torch.Tensor:
    # label: CE on the base labels | teacher: KL to the frozen teacher's softened class distribution | both: their sum
    loss = F.cross_entropy(logits, labels) if target in ("label", "both") else logits.new_zeros((), dtype=torch.float32)
    if target in ("teacher", "both"):
        student = F.log_softmax(logits.float() / tau, dim=-1)
        teacher = F.log_softmax(teacher_logits.float() / tau, dim=-1)
        loss = loss + tau ** 2 * F.kl_div(student, teacher, reduction="batchmean", log_target=True)
    return loss


def parse_cgas_layers(spec: str, n_layers: int) -> list[int]:
    # "all" | "last_K" | a comma list of 1-indexed block numbers -> 0-indexed block positions, ascending
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("last_"):
        k = int(spec[len("last_"):])
        if not 1 <= k <= n_layers:
            raise ValueError(f"model.cgas_layers=last_{k} needs 1 <= k <= {n_layers}")
        return list(range(n_layers - k, n_layers))
    idx = sorted({int(p) - 1 for p in spec.split(",")})
    if not idx or idx[0] < 0 or idx[-1] >= n_layers:
        raise ValueError(f"model.cgas_layers={spec!r} must list block numbers in [1, {n_layers}]")
    return idx


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
    # the student CLIP, the frozen pretrained CLIP (teacher) it is distilled towards and damped from, the projectors

    def __init__(
        self,
        model: CLIP,
        num_frames: int,
        teacher: CLIP | None = None,
        drift_damping: bool = False,
        distill: bool = False,
        alpha: float = 0.1,
        aux: bool = False,
        num_classes: int = 0,
        cgas_source: str | None = None,
        cgas_segments: int = 64,
        cgas_pool_modes: tuple[str, ...] = ("mean",),
        cgas_dropout_p: float = 0.3,
        cgas_ce_weight: float = 1.0,
        cgas_target: str = "label",
        cgas_tau: float = 1.0,
        cgas_layers: str = "all",
        cgas_detach: bool = False,
        vision_dropout: float = 0.0,
    ):
        super().__init__()
        needs_teacher = (
            drift_damping or distill or cgas_source in ("teacher", "both")
            or (cgas_source is not None and cgas_target != "label")
        )
        if needs_teacher and teacher is None:
            raise ValueError(
                "drift damping from the teacher, distilling, cgas_source=teacher/both and cgas_target=teacher/both "
                "need a teacher"
            )
        self.model, self.num_frames, self.drift_damping, self.alpha = model, num_frames, drift_damping, alpha
        self.teacher = teacher.eval().requires_grad_(False) if teacher is not None else None
        self.projector_v = residual_projector(model.embed_dim) if distill else None
        self.projector_t = residual_projector(model.embed_dim) if distill else None
        # reference/KICLIP always builds this alongside distillation and trains it on the pooled, projected embedding
        self.aux = aux_classifier(model.embed_dim, num_classes) if aux else None
        self.cgas_source = cgas_source
        self.cgas_segments, self.cgas_pool_modes = cgas_segments, list(cgas_pool_modes)
        self.cgas_dropout_p, self.cgas_ce_weight = cgas_dropout_p, cgas_ce_weight
        self.cgas_target, self.cgas_tau = cgas_target, cgas_tau
        self.cgas_block_idx = parse_cgas_layers(cgas_layers, len(model.visual.transformer.resblocks))
        self.cgas_detach = cgas_detach
        self.vision_dropout = nn.Dropout(vision_dropout) if vision_dropout > 0 else None
        cgas_channels = 2 if cgas_source == "both" else 1
        # the head's output is its own open-vocabulary classifier (cgas_ce); it is never added back into the
        # video embedding -- the capacity gate only has to shape the backbone's gradient, see docs/cgas.md
        self.cgas_head = (
            CGASHead(len(self.cgas_block_idx), cgas_segments, model.embed_dim, cgas_channels)
            if cgas_source is not None else None
        )

    # the teacher is the pretrained CLIP, rebuilt whenever a model is built, so checkpoints leave it out
    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        return {k: v for k, v in super().state_dict(*args, **kwargs).items() if not k.startswith("teacher.")}

    def load_state_dict(self, state, strict: bool = True, assign: bool = False):
        state = _alias_legacy_damping_keys(state)
        teacher = {k: v for k, v in super().state_dict().items() if k.startswith("teacher.")}
        return super().load_state_dict({**teacher, **state}, strict, assign)

    def encode_video(
        self, video: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor], list[torch.Tensor], torch.Tensor | None]:
        # (..., T, 3, H, W) -> the student's and the teacher's frame features (frames, D), unit norm, every
        # vision block's own output (only when cgas_source is set), and the RAW (unnormalized) CGAS head
        # output, for cgas_ce's own classification loss
        needs_teacher_hidden = self.cgas_source in ("teacher", "both")
        x = video.flatten(0, -4)
        stream = target = None
        student_hidden: list[torch.Tensor] = []
        teacher_hidden: list[torch.Tensor] = []
        if self.teacher is not None:
            with torch.no_grad():
                if needs_teacher_hidden:
                    # teacher_hidden: every block's own output, for CGAS; stream: every block's input (the
                    # embedding, then every output but the last), for drift damping -- same as trace()'s "inputs"
                    embed = self.teacher.visual.embed(x)
                    teacher_hidden = self.teacher.visual.transformer.hidden(embed)
                    stream = [embed] + teacher_hidden[:-1]
                    target = self.teacher.visual.head(teacher_hidden[-1])
                else:
                    target, stream = self.teacher.visual.trace(x)
            target = F.normalize(target, dim=-1)
        if self.cgas_source is not None:
            student_hidden = self.model.visual.hidden(x, stream if self.drift_damping else None)
            frames = self.model.visual.head(student_hidden[-1])
        else:
            frames = self.model.visual(x, stream if self.drift_damping else None)
        cgas_raw = None
        if self.cgas_source is not None:
            if self.training:
                cgas_raw = self._cgas_forward(student_hidden, teacher_hidden, sample_pool_mode(self.cgas_pool_modes))
            else:
                # training resamples a pool mode every step; eval instead averages the head's OUTPUT over
                # every non-dropout mode actually used in training, so eval sees the same mixture of modes
                # the head was trained on, rather than only ever exercising a single ('mean') branch of it
                modes = [m for m in self.cgas_pool_modes if m != "dropout"] or ["mean"]
                cgas_raw = torch.stack(
                    [self._cgas_forward(student_hidden, teacher_hidden, m) for m in modes]
                ).mean(0)
        if self.vision_dropout is not None:
            frames = self.vision_dropout(frames)
        return F.normalize(frames, dim=-1), target, student_hidden, teacher_hidden, cgas_raw

    def _cgas_forward(
        self, student_hidden: list[torch.Tensor], teacher_hidden: list[torch.Tensor], mode: str
    ) -> torch.Tensor:
        # cgas_block_idx: which blocks' CLS feed the head -- all 12 by default, or a subset (model.cgas_layers)
        # to test whether reading across depth matters versus reading only the last block(s)
        # cgas_detach: cgas_ce still trains the head, but sends zero gradient into the backbone, since the
        # tokens it reads are detached first -- isolates "cgas_ce shapes the backbone via its gradient" from
        # cgas.yaml's other channels (see docs/cgas.md)
        channels = []
        if self.cgas_source in ("student", "both"):
            channels.append(torch.stack(
                [build_cgas_input(student_hidden[i].detach() if self.cgas_detach else student_hidden[i],
                                   self.cgas_segments, mode, self.cgas_dropout_p)
                 for i in self.cgas_block_idx], dim=1
            ))
        if self.cgas_source in ("teacher", "both"):
            channels.append(torch.stack(
                [build_cgas_input(teacher_hidden[i], self.cgas_segments, mode, self.cgas_dropout_p)
                 for i in self.cgas_block_idx], dim=1
            ))
        stacked = channels[0] if len(channels) == 1 else torch.cat(channels, dim=-1)  # (N, L, C*M)
        return self.cgas_head(stacked)  # (N, D), unnormalized

    def forward(
        self, video: torch.Tensor, tokens: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # -> logits (B, C), the distillation loss, the aux classifier's CE, the CGAS head's CE
        frames, teacher_frames, student_hidden, teacher_hidden, cgas_raw = self.encode_video(video)  # (B*T, D)
        classes = class_features(self.model, tokens)                                       # (C, D)
        logits = clip_logits(self.model, frames, classes, self.num_frames)
        zero = torch.zeros((), device=logits.device)
        pooled_frames = frames
        kd = zero
        teacher_classes = None
        if self.projector_v is not None or (self.cgas_head is not None and self.cgas_target != "label"):
            with torch.no_grad():
                teacher_classes = class_features(self.teacher, tokens)
        if self.projector_v is not None:
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
        cgas_ce = zero
        if self.cgas_head is not None and labels is not None:
            # F.normalize runs in fp32 under autocast regardless of cgas_raw's dtype; match classes' dtype
            # before the matmul in clip_logits, which autocast does NOT upcast for us here
            cgas_normed = F.normalize(cgas_raw, dim=-1).to(classes.dtype)
            cgas_logits = clip_logits(self.model, cgas_normed, classes, self.num_frames)
            teacher_logits = None
            if self.cgas_target != "label":
                with torch.no_grad():
                    teacher_logits = clip_logits(self.teacher, teacher_frames.to(teacher_classes.dtype), teacher_classes,
                                                 self.num_frames)
            cgas_ce = cgas_loss(cgas_logits, labels, self.cgas_target, self.cgas_tau, teacher_logits)
        return logits, kd, aux, cgas_ce


def build_kiclip(cfg: Config, weights: dict[str, torch.Tensor] | None = None, num_classes: int = 0) -> KICLIP:
    # weights: a trained student to evaluate; None builds a student to train, starting from the pretrained CLIP
    m = cfg.model
    blocks = m.drift_damping and m.drift_damping_source == "blocks"
    from_teacher = m.drift_damping and m.drift_damping_source == "teacher"
    distill = weights is None and m.distill_weight > 0
    aux = weights is None and m.aux_classifier
    cgas_needs_teacher = m.cgas_enabled and m.cgas_source in ("teacher", "both")
    # the teacher target only shapes cgas_ce during training; a checkpoint under test never needs it
    cgas_target = m.cgas_target if weights is None else "label"
    cgas_needs_teacher = cgas_needs_teacher or (m.cgas_enabled and cgas_target != "label")
    pretrained = pretrained_state(m.arch) if weights is None or from_teacher or cgas_needs_teacher else None
    if weights is None:
        weights = with_frozen_copy(pretrained) if blocks else pretrained
    student = CLIP.from_state_dict(
        weights,
        temporal=m.temporal,
        num_frames=cfg.data.num_frames,
        drift_damping=blocks,
        drift_damping_weight=m.drift_damping_weight,
        grad_checkpointing=m.grad_checkpointing,
    )
    teacher = (
        CLIP.from_state_dict(pretrained) if distill or from_teacher or cgas_needs_teacher
        else None
    )
    return KICLIP(
        student, cfg.data.num_frames, teacher, drift_damping=from_teacher, distill=distill, alpha=m.residual_alpha,
        aux=aux, num_classes=num_classes,
        cgas_source=m.cgas_source if m.cgas_enabled else None,
        cgas_segments=m.cgas_segments,
        cgas_pool_modes=tuple(m.cgas_pool_modes),
        cgas_dropout_p=m.cgas_dropout_p,
        cgas_ce_weight=m.cgas_ce_weight,
        cgas_target=cgas_target,
        cgas_tau=m.cgas_tau,
        cgas_layers=m.cgas_layers,
        cgas_detach=m.cgas_detach,
        vision_dropout=m.vision_dropout,
    )
