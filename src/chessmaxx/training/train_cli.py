"""Command-line entry point for the tiny supervised fine-tuning proof."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from chessmaxx.training.config import load_tiny_sft_profile
from chessmaxx.training.train import run_tiny_sft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chessmaxx-train")
    parser.add_argument("--profile", required=True, help="tiny-SFT TOML profile")
    parser.add_argument("--dataset", required=True, help="teacher-labelled JSONL")
    parser.add_argument("--output-dir", required=True, help="run artifacts directory")
    parser.add_argument(
        "--resume-from-checkpoint",
        help="checkpoint directory created by an earlier run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_tiny_sft(
        load_tiny_sft_profile(args.profile),
        profile_path=args.profile,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
