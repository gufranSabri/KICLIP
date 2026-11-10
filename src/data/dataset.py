from __future__ import annotations

import os
import random
from pathlib import Path

import av
import torch
from torch.utils.data import Dataset

from src.config import Config
from src.data.prompts import load_descriptions
from src.data.video import eval_transform, read_clips, train_transform
from src.utils.logger import get_logger

logger = get_logger(__name__)

RETRIES = 10
DECODE_ERRORS = (av.error.FFmpegError, OSError, ValueError)


class VideoDataset(Dataset):
    # train items: (T, 3, S, S) from one random clip; eval items: (V, T, 3, S, S) from `views` evenly spaced clips

    def __init__(self, cfg: Config, split: str, train: bool, views: int = 1):
        d = cfg.data
        self.split, self.train, self.views = split, train, views
        self.root = cfg.video_root(split)
        self.num_frames = d.num_frames
        self.span = d.num_frames * d.sampling_rate / d.target_fps
        self.scales, self.crop, self.flip = (d.train_scales[0], d.train_scales[1]), d.crop_size, d.flip
        self.classnames = load_descriptions(cfg.split_classes(split))
        self.paths, self.labels = self._read(cfg.split_csv(split))

    def _read(self, csv: Path) -> tuple[list[Path], list[int]]:
        rows = [line.rsplit(",", 1) for line in csv.read_text().splitlines() if line.strip()]
        paths = [self.root / p.strip() for p, _ in rows]
        labels = [int(label) for _, label in rows]
        bad = sorted({y for y in labels if not 0 <= y < len(self.classnames)})
        if bad:
            raise ValueError(f"{csv}: labels {bad[:5]} fall outside the {len(self.classnames)} classes of its class file")
        if not self.root.is_dir():
            raise FileNotFoundError(f"no video folder {self.root}; put the videos there or set data_root")

        found = _existing(paths)
        keep = [i for i, p in enumerate(paths) if p in found]
        if len(keep) < len(paths):
            first = next(p for p in paths if p not in found)
            logger.warning("%s: %d/%d videos missing under %s (e.g. %s); they are skipped",
                           self.split, len(paths) - len(keep), len(paths), self.root, first)
        if not keep:
            raise FileNotFoundError(f"{self.split}: none of the {len(paths)} videos exist under {self.root}")
        return [paths[i] for i in keep], [labels[i] for i in keep]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        for _ in range(RETRIES):
            try:
                if self.train:
                    clip = read_clips(self.paths[i], self.num_frames, self.span, None)[0]
                    return train_transform(clip, self.scales, self.crop, self.flip), self.labels[i]
                clips = read_clips(self.paths[i], self.num_frames, self.span, self.views)
                # FROSTER tests at the smallest training scale
                return eval_transform(clips, self.scales[0], self.crop), self.labels[i]
            except DECODE_ERRORS as e:
                if not self.train:
                    raise RuntimeError(f"failed to decode {self.paths[i]}: {e}. Re-encode or replace this video.") from e
                logger.warning("failed to decode %s (%s); sampling another video", self.paths[i], e)
                i = random.randrange(len(self.paths))
        raise RuntimeError(f"{RETRIES} videos in a row failed to decode under {self.root}; check the data")


def _existing(paths: list[Path]) -> set[Path]:
    # one listdir per folder instead of one stat per video
    found: set[Path] = set()
    for parent in {p.parent for p in paths}:
        if parent.is_dir():
            found.update(parent / name for name in os.listdir(parent))
    return found
