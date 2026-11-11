from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

VALID_ARCHS = ("ViT-B-32", "ViT-B-16", "ViT-L-14")
VALID_TEMPORAL = ("expand",)
VALID_INJECT_SOURCES = ("teacher", "blocks")
CROSS_TEST = (
    "cross/ucf101/full", "cross/ucf101/split1", "cross/ucf101/split2", "cross/ucf101/split3",
    "cross/hmdb51/full", "cross/hmdb51/split1", "cross/hmdb51/split2", "cross/hmdb51/split3",
    "cross/k600/split1", "cross/k600/split2", "cross/k600/split3",
)


@dataclass
class ModelConfig:
    arch: str = "ViT-B-16"
    temporal: str | None = "expand"
    inject: bool = True
    inject_source: str = "teacher"
    distill_weight: float = 2.0
    residual_alpha: float = 0.1
    aux_classifier: bool = True
    grad_checkpointing: bool = False


@dataclass
class DataConfig:
    train: str = "cross/k400/train"
    val: str | None = "cross/k400/val"
    test: list[str] = field(default_factory=lambda: list(CROSS_TEST))
    num_frames: int = 8
    sampling_rate: int = 16
    target_fps: float = 30.0
    train_scales: list[int] = field(default_factory=lambda: [224, 256])
    crop_size: int = 224
    flip: bool = True
    num_workers: int = 8


@dataclass
class TrainConfig:
    epochs: int = 22
    batch_size: int = 4  # global; divided across GPUs, no gradient accumulation, as in reference/KICLIP
    lr: float = 3.33e-6
    warmup_epochs: float = 2.0
    warmup_start_lr: float = 3.33e-8
    end_lr: float = 3.33e-8
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True
    average_from: int = 2
    eval_every: int = 0


@dataclass
class TestConfig:
    views: int = 3
    batch_size: int = 8
    amp: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    test: TestConfig = field(default_factory=TestConfig)
    seed: int = 0
    data_root: str = "~/scratch/ahmedubc/KICLIP-data"
    splits_root: str = "splits"
    runs_root: str = "runs"

    def split_csv(self, split: str) -> Path:
        return REPO_ROOT / Path(self.splits_root).expanduser() / f"{split}.csv"

    def split_classes(self, split: str) -> Path:
        own = self.split_csv(split).with_suffix(".json")
        return own if own.exists() else own.parent / "classes.json"

    def video_root(self, split: str) -> Path:
        return Path(self.data_root).expanduser() / split.split("/")[1]

    def run_dir(self, run_name: str | None = None) -> Path:
        return Path(self.runs_root).expanduser() / (run_name or "_".join(self.data.train.split("/")))

    def check_split(self, split: str, key: str) -> None:
        if len(split.split("/")) != 3:
            raise ValueError(f"{key}: split {split!r} must look like <protocol>/<dataset>/<set>, e.g. b2n/k400/base")
        for path in (self.split_csv(split), self.split_classes(split)):
            if not path.exists():
                raise ValueError(f"{key}: {split!r} needs {path}, which does not exist")

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        m, d, t, te = self.model, self.data, self.train, self.test

        if m.arch not in VALID_ARCHS:
            raise ValueError(f"model.arch must be one of {VALID_ARCHS}, got {m.arch!r}")
        if m.temporal is not None and m.temporal not in VALID_TEMPORAL:
            raise ValueError(f"model.temporal must be one of {VALID_TEMPORAL} or null, got {m.temporal!r}")
        if m.inject_source not in VALID_INJECT_SOURCES:
            raise ValueError(f"model.inject_source must be one of {VALID_INJECT_SOURCES}, got {m.inject_source!r}")
        if m.distill_weight < 0:
            raise ValueError("model.distill_weight must be >= 0")
        if m.residual_alpha < 0:
            raise ValueError("model.residual_alpha must be >= 0")

        self.check_split(d.train, "data.train")
        if d.val is not None:
            self.check_split(d.val, "data.val")
        if not d.test:
            raise ValueError("data.test must list at least one split")
        for split in d.test:
            self.check_split(split, "data.test")
        if d.num_frames < 1 or d.sampling_rate < 1:
            raise ValueError("data.num_frames and data.sampling_rate must be >= 1")
        if d.target_fps <= 0:
            raise ValueError("data.target_fps must be > 0")
        if len(d.train_scales) != 2 or d.train_scales[0] > d.train_scales[1]:
            raise ValueError(f"data.train_scales must be [low, high] with low <= high, got {d.train_scales}")
        if d.crop_size > d.train_scales[0]:
            raise ValueError(f"data.crop_size ({d.crop_size}) must be <= data.train_scales[0] ({d.train_scales[0]})")
        if d.num_workers < 0:
            raise ValueError("data.num_workers must be >= 0")

        if t.epochs < 1:
            raise ValueError("train.epochs must be >= 1")
        if t.batch_size < 1:
            raise ValueError("train.batch_size must be >= 1")
        if not 0 <= t.warmup_epochs < t.epochs:
            raise ValueError(f"train.warmup_epochs ({t.warmup_epochs}) must be in [0, train.epochs={t.epochs})")
        if not 0 < t.end_lr < t.lr:
            raise ValueError(f"train.end_lr ({t.end_lr}) must be in (0, train.lr={t.lr})")
        if not 1 <= t.average_from <= t.epochs:
            raise ValueError(f"train.average_from ({t.average_from}) must be in [1, train.epochs={t.epochs}]")
        if t.eval_every < 0:
            raise ValueError("train.eval_every must be >= 0")
        if t.eval_every and d.val is None:
            raise ValueError("train.eval_every > 0 needs data.val")

        if te.views < 1 or te.batch_size < 1:
            raise ValueError("test.views and test.batch_size must be >= 1")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError(f"seed must be an integer >= 0, got {self.seed!r}")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


_SECTIONS = {
    "model": ModelConfig,
    "data": DataConfig,
    "train": TrainConfig,
    "test": TestConfig,
}
_TOP = ("seed", "data_root", "splits_root", "runs_root")


def _coerce(cls, section: dict) -> dict:
    # YAML reads exponents without a dot (1e-7) as strings
    floats = {f.name for f in fields(cls) if f.type == "float"}
    return {
        k: float(v) if k in floats and isinstance(v, (int, str)) and not isinstance(v, bool) else v
        for k, v in section.items()
    }


def _build(raw: dict) -> Config:
    kwargs: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        section = raw.get(name) or {}
        if not isinstance(section, dict):
            raise ValueError(f"config section '{name}' must be a mapping")
        known = {f.name for f in fields(cls)}
        unknown = set(section) - known
        if unknown:
            raise ValueError(f"unknown key(s) in '{name}': {sorted(unknown)}. Valid: {sorted(known)}")
        kwargs[name] = cls(**_coerce(cls, section))

    for top in _TOP:
        if top in raw:
            kwargs[top] = raw[top]

    unknown_top = set(raw) - set(_SECTIONS) - set(_TOP)
    if unknown_top:
        raise ValueError(f"unknown top-level config key(s): {sorted(unknown_top)}")

    cfg = Config(**kwargs)
    cfg.validate()
    return cfg


def parse_overrides(items: list[str]) -> dict:
    # --set a.b=c -> {"a": {"b": c}}, with c parsed as YAML so numbers, bools, lists and null come through typed
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def load_config(paths: str | Path | list[str | Path], overrides: dict | None = None) -> Config:
    # default.yaml, then each file in order (e.g. a dataset config, then a method overlay), then --set
    paths = [Path(p) for p in (paths if isinstance(paths, list) else [paths])]
    raw: dict = {}
    if DEFAULT_CONFIG.exists():
        with open(DEFAULT_CONFIG) as f:
            raw = yaml.safe_load(f) or {}

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        with open(path) as f:
            raw = _deep_merge(raw, yaml.safe_load(f) or {})

    if overrides:
        raw = _deep_merge(raw, overrides)

    return _build(raw)
