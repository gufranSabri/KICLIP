# A run's identity comes from its own config.json (data.train), never its directory name.

from __future__ import annotations

import json
import statistics
from pathlib import Path


def load_runs(runs_root: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            continue

        runs.append({
            "run": run_dir.name,
            "train": cfg["data"]["train"],
            "arch": cfg["model"]["arch"],
            "method": method(cfg),
            "done": (run_dir / "averaged.pt").exists(),
            "tests": load_tests(run_dir / "results.json", cfg["data"]["train"]),
        })
    return runs


def load_tests(path: Path, train: str) -> dict[str, dict]:
    # results.json keys (base, novel, <dataset>/<set>) back to full split names; HM is recomputed by the tables
    if not path.exists():
        return {}
    try:
        results = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    protocol, dataset = train.split("/")[:2]
    return {
        f"{protocol}/{key}" if "/" in key else f"{protocol}/{dataset}/{key}": m
        for key, m in results.items() if isinstance(m, dict) and set(m) == {"top1", "top5"}
    }


METHODS = ("kiclip_aux", "kiclip", "kiclip_blocks", "froster")
METHOD_PRETTY = {
    "kiclip_aux": "KICLIP + aux", "kiclip": "KICLIP", "kiclip_blocks": "KICLIP (blocks)", "froster": "FROSTER",
}


def method(cfg: dict) -> str:
    # configs from before model.drift_damping_source damped from frozen block copies / model.aux_classifier existed
    if not cfg["model"].get("drift_damping", False):
        base = "froster"
    else:
        base = "kiclip" if cfg["model"].get("drift_damping_source", "blocks") == "teacher" else "kiclip_blocks"
    return f"{base}_aux" if cfg["model"].get("aux_classifier", False) else base


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def fmt(x: float | None, std: float | None = None) -> str:
    if x is None:
        return "-"
    return f"{x:.1f}" if std is None else f"{x:.1f} ± {std:.1f}"
