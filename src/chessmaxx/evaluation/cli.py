"""Command-line entry point for the Chessmaxx evaluation harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from chessmaxx import __version__
from chessmaxx.evaluation.dataset import load_positions
from chessmaxx.evaluation.model import HuggingFaceMoveGenerator
from chessmaxx.evaluation.runner import EvaluationRunner
from chessmaxx.evaluation.stockfish import StockfishAnalyzer, StockfishConfig


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chessmaxx-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    positions = subparsers.add_parser(
        "positions", help="evaluate a model on a frozen JSONL position set"
    )
    positions.add_argument("--model", required=True)
    positions.add_argument("--dataset", type=Path, required=True)
    positions.add_argument("--output", type=Path, required=True)
    positions.add_argument("--stockfish", default="stockfish")
    positions.add_argument("--cache", type=Path, default=Path("artifacts/cache.json"))
    positions.add_argument("--device")
    positions.add_argument("--batch-size", type=_positive_int, default=8)
    positions.add_argument("--max-new-tokens", type=_positive_int, default=8)
    positions.add_argument("--nodes", type=_positive_int, default=50_000)
    positions.add_argument("--multipv", type=_positive_int, default=3)
    positions.add_argument("--threads", type=_positive_int, default=1)
    positions.add_argument("--hash-mb", type=_positive_int, default=64)
    positions.add_argument(
        "--limit", type=_positive_int, help="evaluate only the first N positions"
    )
    return parser


def run_positions(args: argparse.Namespace) -> int:
    positions = load_positions(args.dataset)
    if args.limit is not None:
        positions = positions[: args.limit]
    model = HuggingFaceMoveGenerator.from_pretrained(
        args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    engine_config = StockfishConfig(
        nodes=args.nodes,
        multipv=args.multipv,
        threads=args.threads,
        hash_mb=args.hash_mb,
    )
    settings = {
        "chessmaxx_version": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "dataset_path": str(args.dataset),
        "dataset_sha256": _sha256(args.dataset),
        "position_limit": args.limit,
        "stockfish": asdict(engine_config),
    }
    with StockfishAnalyzer.open(
        args.stockfish, config=engine_config, cache_path=args.cache
    ) as analyzer:
        report = EvaluationRunner(
            model,
            analyzer,
            batch_size=args.batch_size,
            settings=settings,
        ).run(positions)
    report.write(args.output)
    print(json.dumps(report.summary, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "positions":
        return run_positions(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

