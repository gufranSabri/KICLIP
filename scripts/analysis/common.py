# Shared checkpoint-loading helper for the DCGD analysis scripts.

from __future__ import annotations

import sys
from dataclasses import fields, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.config import Config, ModelConfig, load_config  # noqa: E402
from src.models.kiclip import KICLIP, build_kiclip  # noqa: E402
from src.test import load_weights  # noqa: E402


def load_checkpoint(spec: str, cfg: Config, device: torch.device) -> tuple[KICLIP, dict]:
    # spec: a run dir or an averaged.pt path; builds a KICLIP with that checkpoint's own model config
    path = Path(spec) / "averaged.pt" if Path(spec).is_dir() else Path(spec)
    state, ckpt_cfg = load_weights(path)
    if ckpt_cfg is None:
        raise ValueError(f"{path} carries no config; this script needs an averaged.pt written by src/train.py")
    known = {f.name for f in fields(ModelConfig)}
    model_cfg = ModelConfig(**{k: v for k, v in ckpt_cfg["model"].items() if k in known})
    weights = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    model = build_kiclip(replace(cfg, model=model_cfg), weights)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), ckpt_cfg
