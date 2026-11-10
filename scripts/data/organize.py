#!/usr/bin/env python3
# Moves downloaded videos from {data_root}/raw/<dataset> to {data_root}/<dataset> under the names the split CSVs use.
#
#     bash scripts/data/download.sh
#     python scripts/data/organize.py --check --workers 8                 # every dataset with a raw/ folder
#     python scripts/data/organize.py --datasets k400 --check --delete-raw

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config, parse_overrides  # noqa: E402
from src.utils.progress import progress  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".webm"}
KINETICS = {"k400", "k600"}
STAMP = re.compile(r"_\d{6}_\d{6}$")
ORGANIZED = ".organized"


def video_key(name: str, dataset: str) -> str:
    # Kinetics files are matched by YouTube id: CVDF names them <id>_<start>_<end>.mp4, the val CSVs <id>.mp4/.mkv
    stem = Path(name).stem
    return STAMP.sub("", stem) if dataset in KINETICS else stem


def index_raw(raw: Path, dataset: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for dirpath, dirnames, files in os.walk(raw):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for f in files:
            if Path(f).suffix.lower() in VIDEO_EXT:
                found.setdefault(video_key(f, dataset), Path(dirpath) / f)
    return found


def probe(path: str) -> str | None:
    # PyAV only: importing torch in every worker is what gets the pool killed on a login node
    import av

    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            if not (stream.duration or container.duration or stream.frames):
                return "no duration in the container metadata"
            for _ in container.decode(stream):
                return None
        return "no decodable frames"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def check_videos(paths: list[Path], cache: Path, workers: int) -> set[str]:
    # results are cached in raw/, flushed as they come in, so an interrupted check resumes where it stopped
    done = json.loads(cache.read_text()) if cache.exists() else {}
    todo = sorted({str(p) for p in paths} - set(done))
    if todo:
        with ProcessPoolExecutor(workers) as pool:
            results = zip(todo, pool.map(probe, todo, chunksize=64))
            for i, (path, err) in enumerate(progress(results, total=len(todo), desc=f"checking {cache.parent.name}",
                                                     ncols=100)):
                done[path] = err
                if i % 5000 == 4999:
                    cache.write_text(json.dumps(done))
        cache.write_text(json.dumps(done))
    return {p for p in map(str, paths) if done.get(p)}


def settled(dest: Path, raw: Path) -> bool:
    return dest.exists() and raw not in dest.resolve().parents


def place(src: Path, dests: list[Path]) -> None:
    # the first CSV name gets the file itself; any other name for the same video is a hard link to it
    first, *rest = dests
    for d in dests:
        if d.is_symlink():
            d.unlink()
        d.parent.mkdir(parents=True, exist_ok=True)
    if not first.exists():
        os.replace(src, first)
    for d in rest:
        if not d.exists():
            try:
                os.link(first, d)
            except OSError:
                d.symlink_to(os.path.relpath(first, d.parent))


def organize(ds: str, csvs: list[Path], splits_root: Path, data_root: Path, args) -> list[tuple]:
    raw, out = data_root / "raw" / ds, data_root / ds
    if (out / ORGANIZED).exists() and not raw.is_dir():
        print(f"  [skip] {ds}: already organized under {out}")
        return []
    if not raw.is_dir():
        print(f"  [skip] {ds}: no {raw} (run scripts/data/download.sh {ds})")
        return []

    print(f"  indexing {raw}")
    found = index_raw(raw, ds)
    rels = {csv: [ln.rsplit(",", 1)[0].strip() for ln in csv.read_text().splitlines() if ln.strip()] for csv in csvs}
    dests = {out / r for rs in rels.values() for r in rs}

    pending: dict[Path, list[Path]] = defaultdict(list)
    for dest in sorted(dests):
        src = found.get(video_key(dest.name, ds))
        if not settled(dest, raw) and src is not None:
            pending[src].append(dest)

    broken = check_videos(list(pending), raw / ".probe.json", args.workers) if args.check else set()
    for src, ds_dests in progress(pending.items(), desc=f"moving {ds}", ncols=100):
        if str(src) in broken:
            for d in ds_dests:
                if d.is_symlink():
                    d.unlink()
        else:
            place(src, ds_dests)

    logs = data_root / "logs"
    logs.mkdir(exist_ok=True)
    report = []
    for csv, rs in rels.items():
        split = "/".join(csv.relative_to(splits_root).with_suffix("").parts)
        bad = [r for r in rs if not settled(out / r, raw) and str(found.get(video_key(Path(r).name, ds))) in broken]
        missing = [r for r in rs if not settled(out / r, raw) and r not in bad]
        log = logs / f"unavailable_{split.replace('/', '_')}.txt"
        if missing or bad:
            log.write_text("".join(f"missing {r}\n" for r in missing) + "".join(f"broken  {r}\n" for r in bad))
        else:
            log.unlink(missing_ok=True)
        report.append((split, len(rs), len(rs) - len(missing) - len(bad), len(missing), len(bad)))

    if args.delete_raw:
        shutil.rmtree(raw)
        (out / ORGANIZED).touch()
        print(f"  deleted {raw}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="move downloaded videos into the layout the split CSVs expect")
    ap.add_argument("--datasets", nargs="+", default=None, help="default: every dataset with a raw/ folder")
    ap.add_argument("--check", action="store_true", help="decode a frame of every video and leave out broken ones")
    ap.add_argument("--delete-raw", action="store_true", help="delete raw/<dataset> once its videos are moved")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", action="append", default=[], help="config(s) giving data_root and splits_root")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    data_root = Path(cfg.data_root).expanduser().resolve()
    splits_root = REPO_ROOT / Path(cfg.splits_root).expanduser()

    by_dataset: dict[str, list[Path]] = defaultdict(list)
    for csv in sorted(splits_root.glob("*/*/*.csv")):
        by_dataset[csv.parent.name].append(csv)

    report = []
    for ds in args.datasets or sorted(by_dataset):
        report += organize(ds, by_dataset[ds], splits_root, data_root, args)

    print()
    print(f"  {'split':28s} {'videos':>7s} {'ready':>7s} {'missing':>8s} {'broken':>7s}   coverage")
    for split, n, ok, miss, bad in report:
        print(f"  {split:28s} {n:7d} {ok:7d} {miss:8d} {bad:7d}   {100 * ok / max(n, 1):6.2f}%")
    print(f"\n  videos under {data_root}/<dataset>; unavailable ones per split in {data_root / 'logs'}")


if __name__ == "__main__":
    main()
