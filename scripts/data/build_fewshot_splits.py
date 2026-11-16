#!/usr/bin/env python3
# One-time conversion of ViFi-CLIP's few-shot and zero-shot-val split files (the de facto standard
# OST/MoTE/MSTA all reuse) into this repo's splits/<protocol>/<dataset>/<set>.csv format. Source files
# are plain "<filename> <label_idx>" text; this remaps every label through CLASS NAME against our own
# classes.json, since ViFi-CLIP's label ordering is not always identical (e.g. HMDB51 swaps kick/kick
# ball and sword/sword exercise at indices 20/21, 44/45). Scoped to HMDB51 and UCF101 (no SSv2).
#
#     python scripts/data/build_fewshot_splits.py --vifi-dir /path/to/fetched/vifi/files

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

SPLITS_ROOT = REPO_ROOT / "splits"

# dataset -> (vifi split-file prefix, vifi label csv name, our cross/<dataset> classes.json, has class folders)
DATASETS = {
    "hmdb51": ("hmdb_splits", "hmdb_51_labels.csv", True),
    "ucf101": ("ucf_splits", "ucf_101_labels.csv", True),
}
SHOTS = (2, 4, 8, 16)
VAL_SPLITS = (1, 2, 3)


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def our_classes(dataset: str) -> dict[str, str]:
    # cross/<dataset>/classes.json: "idx" -> "Name: description"; we only need the Name
    path = SPLITS_ROOT / "cross" / dataset / "classes.json"
    raw = json.loads(path.read_text())
    return {idx: text.split(":")[0].strip() for idx, text in raw.items()}


def vifi_labels(vifi_dir: Path, csv_name: str) -> dict[str, str]:
    # vifi label csv: "id,name" header + rows; id is a 0-indexed string matching their split files' labels
    lines = (vifi_dir / "labels" / csv_name).read_text().splitlines()[1:]
    out = {}
    for line in lines:
        idx, name = line.split(",", 1)
        out[idx] = name.strip().strip('"')
    return out


def build_remap(our: dict[str, str], theirs: dict[str, str]) -> dict[str, str]:
    # their label idx -> our label idx, matched by normalized class name
    by_name = {norm(name): idx for idx, name in our.items()}
    remap = {}
    for their_idx, their_name in theirs.items():
        our_idx = by_name.get(norm(their_name))
        if our_idx is None:
            raise ValueError(f"no match for {their_name!r} (their idx {their_idx}) in our classes.json")
        remap[their_idx] = our_idx
    return remap


def class_folder_map(data_root: Path, dataset: str) -> dict[str, str]:
    # filename -> its class-folder name, scanned from the real data directory (ViFi-CLIP's split files
    # give filename + label only, not the folder path our CSVs need)
    root = data_root / dataset
    out = {}
    for class_dir in root.iterdir():
        if class_dir.is_dir():
            for f in class_dir.iterdir():
                out[f.name] = class_dir.name
    return out


def convert(src: Path, dst: Path, remap: dict[str, str], folders: dict[str, str] | None) -> tuple[int, int]:
    kept, skipped = 0, 0
    rows = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        fname, their_idx = line.rsplit(" ", 1)
        our_idx = remap[their_idx]
        if folders is not None:
            folder = folders.get(fname)
            if folder is None:
                skipped += 1
                continue
            rows.append(f"{folder}/{fname},{our_idx}")
        else:
            rows.append(f"{fname},{our_idx}")
        kept += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(rows) + "\n")
    return kept, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vifi-dir", required=True, help="local directory holding ViFi-CLIP's datasets_splits/ and labels/")
    ap.add_argument("--data-root", default="~/scratch/ahmedubc/KICLIP-data", help="for scanning class-folder names")
    args = ap.parse_args()

    vifi_dir = Path(args.vifi_dir)
    data_root = Path(args.data_root).expanduser()

    for dataset, (split_dir, labels_csv, has_folders) in DATASETS.items():
        our = our_classes(dataset)
        theirs = vifi_labels(vifi_dir, labels_csv)
        remap = build_remap(our, theirs)
        folders = class_folder_map(data_root, dataset) if has_folders else None
        out_dir = SPLITS_ROOT / "fewshot" / dataset

        for k in SHOTS:
            src = vifi_dir / "datasets_splits" / split_dir / f"train1_few_shot_{k}.txt"
            kept, skipped = convert(src, out_dir / f"train_k{k}.csv", remap, folders)
            print(f"  {dataset} train_k{k}: {kept} kept, {skipped} skipped (not found under {data_root / dataset})")
        for v in VAL_SPLITS:
            src = vifi_dir / "datasets_splits" / split_dir / f"val{v}.txt"
            kept, skipped = convert(src, out_dir / f"val{v}.csv", remap, folders)
            print(f"  {dataset} val{v}: {kept} kept, {skipped} skipped (not found under {data_root / dataset})")

        (out_dir / "classes.json").write_text((SPLITS_ROOT / "cross" / dataset / "classes.json").read_text())
    print(f"wrote {SPLITS_ROOT}/fewshot/<dataset>/*.csv")


if __name__ == "__main__":
    main()
