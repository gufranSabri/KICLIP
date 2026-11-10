from __future__ import annotations

import json
from pathlib import Path

import torch

TEMPLATES = (
    "a photo of {}.",
    "a photo of a person {}.",
    "a photo of a person using {}.",
    "a photo of a person doing {}.",
    "a photo of a person during {}.",
    "a photo of a person performing {}.",
    "a photo of a person practicing {}.",
    "a video of {}.",
    "a video of a person {}.",
    "a video of a person using {}.",
    "a video of a person doing {}.",
    "a video of a person during {}.",
    "a video of a person performing {}.",
    "a video of a person practicing {}.",
    "a example of {}.",
    "a example of a person {}.",
    "a example of a person using {}.",
    "a example of a person doing {}.",
    "a example of a person during {}.",
    "a example of a person performing {}.",
    "a example of a person practicing {}.",
    "a demonstration of {}.",
    "a demonstration of a person {}.",
    "a demonstration of a person using {}.",
    "a demonstration of a person doing {}.",
    "a demonstration of a person during {}.",
    "a demonstration of a person performing {}.",
    "a demonstration of a person practicing {}.",
)


def load_descriptions(path: Path) -> list[str]:
    raw = json.loads(Path(path).read_text())
    if set(raw) != {str(i) for i in range(len(raw))}:
        raise ValueError(f"{path}: class ids must be the strings '0'..'{len(raw) - 1}'")
    return [raw[str(i)] for i in range(len(raw))]


def prompts(descriptions: list[str]) -> list[list[str]]:
    # "Name: description" -> every template filled with Name and followed by the description; the bare description comes last
    names = [d.split(":")[0] for d in descriptions]
    rows = [[t.format(n) + " " + d for n, d in zip(names, descriptions)] for t in TEMPLATES]
    return rows + [list(descriptions)]


def tokenize(descriptions: list[str], context_length: int = 77) -> torch.Tensor:
    import open_clip

    return torch.stack([open_clip.tokenize(row, context_length=context_length) for row in prompts(descriptions)])
