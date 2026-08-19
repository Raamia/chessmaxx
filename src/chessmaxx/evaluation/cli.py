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
from chessmaxx.training.config import load_tiny_sft_profile
from chessmaxx.training.dataset import (
    evaluation_positions_for_split,
    load_training_examples,
)


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


def _tree_sha256(path: Path) -> str:
    root = path.resolve()
    if not root.is_dir():
        raise ValueError(f"adapter directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"adapter directory contains no files: {root}")
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with item.open("rb") as handle:
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
    positions.add_argument("--journal", type=Path)
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
    baseline.add_argument(
        "--journal",
        type=Path,
        help="progress journal path (defaults beside the final report)",
    )
    baseline.add_argument("--device", help="override the device in the profile")
    baseline.add_argument(
        "--limit", type=_positive_int, help="evaluate only the first N positions"
    )

    adapter = subparsers.add_parser(
        "adapter", help="evaluate a trained PEFT adapter on a labelled split"
    )
    adapter.add_argument("--training-profile", type=Path, required=True)
    adapter.add_argument("--adapter-dir", type=Path, required=True)
    adapter.add_argument("--dataset", type=Path, required=True)
    adapter.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    adapter.add_argument("--output", type=Path, required=True)
    adapter.add_argument("--stockfish", default="stockfish")
    adapter.add_argument("--cache", type=Path, default=Path("artifacts/cache.json"))
    adapter.add_argument("--journal", type=Path)
    adapter.add_argument("--device")
    adapter.add_argument("--batch-size", type=_positive_int, default=8)
    adapter.add_argument("--max-new-tokens", type=_positive_int, default=8)
    adapter.add_argument("--nodes", type=_positive_int, default=50_000)
    adapter.add_argument("--multipv", type=_positive_int, default=3)
    adapter.add_argument("--threads", type=_positive_int, default=1)
    adapter.add_argument("--hash-mb", type=_positive_int, default=64)
    adapter.add_argument(
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
        journal_path = args.journal or args.output.with_suffix(
            args.output.suffix + ".progress.jsonl"
        )
        report = EvaluationRunner(
            model,
            analyzer,
            batch_size=args.batch_size,
            settings=settings,
            journal_path=journal_path,
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
        journal_path = args.journal or args.output.with_suffix(
            args.output.suffix + ".progress.jsonl"
        )
        report = EvaluationRunner(
            model,
            analyzer,
            batch_size=profile.batch_size,
            settings=settings,
            journal_path=journal_path,
        ).run(positions)
    report.write(args.output)
    print(json.dumps(report.summary, indent=2, sort_keys=True))
    return 0


def run_adapter(args: argparse.Namespace) -> int:
    profile = load_tiny_sft_profile(args.training_profile)
    positions = evaluation_positions_for_split(
        load_training_examples(args.dataset), args.split
    )
    if args.limit is not None:
        positions = positions[: args.limit]
    model = HuggingFaceMoveGenerator.from_adapter(
        args.adapter_dir,
        base_model_name=profile.model_id,
        revision=profile.revision,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        dtype=profile.dtype,
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
        "training_profile_path": str(args.training_profile),
        "training_profile_sha256": _sha256(args.training_profile),
        "training_profile": asdict(profile),
        "adapter_path": str(args.adapter_dir.resolve()),
        "adapter_sha256": _tree_sha256(args.adapter_dir),
        "split": args.split,
        "position_limit": args.limit,
        "stockfish": asdict(engine_config),
        "mode": "adapter",
    }
    with StockfishAnalyzer.open(
        args.stockfish, config=engine_config, cache_path=args.cache
    ) as analyzer:
        journal_path = args.journal or args.output.with_suffix(
            args.output.suffix + ".progress.jsonl"
        )
        report = EvaluationRunner(
            model,
            analyzer,
            batch_size=args.batch_size,
            settings=settings,
            journal_path=journal_path,
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
    if args.command == "adapter":
        return run_adapter(args)
    if args.command == "sample-pgn":
        return run_sample_pgn(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
