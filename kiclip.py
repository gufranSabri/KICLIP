#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from src.config import load_config, parse_overrides
from src.utils.logger import setup_logging

DESCRIPTION = """KICLIP: knowledge injection on top of FROSTER for open-vocabulary action recognition.

subcommands:
    train   fine-tune CLIP on data.train; writes <run>/averaged.pt
    test    evaluate a checkpoint on data.test (or --splits)

--config can be repeated; later files override earlier ones, e.g.
    --config configs/b2n/hmdb51.yaml --config configs/methods/froster.yaml
"""


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", action="append", default=[],
        help="YAML config(s) merged over configs/default.yaml in order, e.g. configs/b2n/k400.yaml",
    )
    common.add_argument(
        "--set", action="append", metavar="KEY=VAL", default=[],
        help="override any config key, e.g. --set train.epochs=12 --set model.inject=false",
    )
    parser = argparse.ArgumentParser(
        prog="kiclip.py",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", parents=[common], help="train on data.train, write <run>/averaged.pt")
    p_tr.add_argument("--run-name", default=None, help="defaults to data.train with '/' -> '_'")

    p_te = sub.add_parser("test", parents=[common], help="evaluate a checkpoint")
    p_te.add_argument("--checkpoint", required=True, help="averaged.pt from train, or an original FROSTER / KICLIP .pth")
    p_te.add_argument(
        "--splits", nargs="+", default=None,
        help="splits to evaluate, e.g. cross/ucf101/split1 b2n/k400/novel (default: data.test)",
    )

    # the subparser's defaults would otherwise overwrite flags given before the subcommand
    pre, _ = common.parse_known_args(argv)
    args = parser.parse_args(argv)
    args.config = list(pre.config) + [c for c in args.config if c not in pre.config]
    args.set = list(pre.set) + [s for s in args.set if s not in pre.set]

    if not args.config:
        parser.error("--config is required (before or after the subcommand)")
    setup_logging()

    try:
        overrides = parse_overrides(args.set)
    except ValueError as e:
        parser.error(str(e))
    cfg = load_config(args.config, overrides=overrides)

    if args.cmd == "train":
        from src.train import train

        train(cfg, run_name=args.run_name)

    elif args.cmd == "test":
        from src.test import test

        test(cfg, args.checkpoint, splits=args.splits)

    return 0


if __name__ == "__main__":
    sys.exit(main())
