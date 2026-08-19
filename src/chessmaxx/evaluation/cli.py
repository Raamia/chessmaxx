"""Command-line entry point for the Chessmaxx evaluation harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from chessmaxx import __version__
from chessmaxx.evaluation.config import load_baseline_profile
from chessmaxx.evaluation.dataset import load_positions
from chessmaxx.evaluation.dataset import write_positions
from chessmaxx.evaluation.model import HuggingFaceMoveGenerator
from chessmaxx.evaluation.runner import EvaluationRunner
from chessmaxx.evaluation.sampling import sample_pgn_positions
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
    positions.add_argument("--revision", default="main")
    positions.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
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

    baseline = subparsers.add_parser(
        "baseline", help="evaluate a model using a checked-in baseline profile"
    )
    baseline.add_argument("--profile", type=Path, required=True)
    baseline.add_argument("--dataset", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--stockfish", default="stockfish")
    baseline.add_argument("--cache", type=Path, default=Path("artifacts/cache.json"))
    baseline.add_argument("--device", help="override the device in the profile")
    baseline.add_argument(
        "--limit", type=_positive_int, help="evaluate only the first N positions"
    )

    sample = subparsers.add_parser(
        "sample-pgn", help="create a deterministic frozen position set from PGN"
    )
    sample.add_argument("--pgn", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--count", type=_positive_int, required=True)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--minimum-ply", type=int, default=8)
    sample.add_argument("--max-per-game", type=_positive_int, default=4)
    return parser


def run_positions(args: argparse.Namespace) -> int:
    positions = load_positions(args.dataset)
    if args.limit is not None:
        positions = positions[: args.limit]
    model = HuggingFaceMoveGenerator.from_pretrained(
        args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        revision=args.revision,
        dtype=args.dtype,
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
        "mode": "positions",
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


def run_baseline(args: argparse.Namespace) -> int:
    profile = load_baseline_profile(args.profile)
    positions = load_positions(args.dataset)
    if args.limit is not None:
        positions = positions[: args.limit]
    selected_device = args.device or profile.device
    model = HuggingFaceMoveGenerator.from_pretrained(
        profile.model_id,
        device=selected_device,
        max_new_tokens=profile.max_new_tokens,
        revision=profile.revision,
        dtype=profile.dtype,
    )
    engine_config = StockfishConfig(
        nodes=profile.stockfish_nodes,
        multipv=profile.stockfish_multipv,
        threads=profile.stockfish_threads,
        hash_mb=profile.stockfish_hash_mb,
    )
    settings = {
        "chessmaxx_version": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "dataset_path": str(args.dataset),
        "dataset_sha256": _sha256(args.dataset),
        "position_limit": args.limit,
        "stockfish": asdict(engine_config),
        "mode": "baseline",
        "profile_path": str(args.profile),
        "profile_sha256": _sha256(args.profile),
        "profile": asdict(profile),
        "effective_device": selected_device,
    }
    with StockfishAnalyzer.open(
        args.stockfish, config=engine_config, cache_path=args.cache
    ) as analyzer:
        report = EvaluationRunner(
            model,
            analyzer,
            batch_size=profile.batch_size,
            settings=settings,
        ).run(positions)
    report.write(args.output)
    print(json.dumps(report.summary, indent=2, sort_keys=True))
    return 0


def run_sample_pgn(args: argparse.Namespace) -> int:
    positions = sample_pgn_positions(
        args.pgn,
        args.count,
        seed=args.seed,
        minimum_ply=args.minimum_ply,
        max_per_game=args.max_per_game,
    )
    write_positions(args.output, positions)
    summary = {
        "positions": len(positions),
        "games": len({position.game_id for position in positions}),
        "phases": dict(Counter(position.phase for position in positions)),
        "output": str(args.output),
        "sha256": _sha256(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "positions":
        return run_positions(args)
    if args.command == "baseline":
        return run_baseline(args)
    if args.command == "sample-pgn":
        return run_sample_pgn(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
