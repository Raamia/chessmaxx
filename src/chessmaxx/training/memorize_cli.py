"""CLI for the deliberately in-sample tiny-SFT memorization check."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from chessmaxx.training.config import load_tiny_sft_profile
from chessmaxx.training.memorize import run_memorization_check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chessmaxx-memorize")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_memorization_check(
        load_tiny_sft_profile(args.profile),
        profile_path=args.profile,
        dataset_path=args.dataset,
        adapter_dir=args.adapter_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
