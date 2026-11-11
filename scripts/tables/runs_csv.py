#!/usr/bin/env python3
# Dumps every run under runs/ as its own CSV row, base/novel/hm, no averaging across seeds.
#
#     python scripts/tables/runs_csv.py

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.tables.collect_runs import load_runs  # noqa: E402
from src.utils.metrics import harmonic_mean  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=str(REPO_ROOT / "runs"))
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "tables" / "runs.csv"))
    args = ap.parse_args()

    runs_root, out_path = Path(args.runs_root), Path(args.out)
    if not runs_root.is_dir():
        raise SystemExit(f"no runs directory at {runs_root}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in load_runs(runs_root):
        protocol, dataset = r["train"].split("/")[:2]
        base = r["tests"].get(f"{protocol}/{dataset}/base")
        novel = r["tests"].get(f"{protocol}/{dataset}/novel")
        hm = harmonic_mean(base["top1"], novel["top1"]) if base and novel else None
        rows.append([
            r["run"], dataset, r["method"], r["train"], r["done"],
            base["top1"] if base else None, novel["top1"] if novel else None, hm,
        ])
        print(f"  {r['run']:20s} {dataset:10s} {r['method']:14s} done={r['done']}")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "dataset", "method", "train", "done", "base", "novel", "hm"])
        for row in rows:
            w.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
