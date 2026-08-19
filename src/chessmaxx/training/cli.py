"""Command-line workflow for building Stockfish-supervised datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from chessmaxx.evaluation.sampling import sample_pgn_positions
from chessmaxx.evaluation.stockfish import StockfishAnalyzer, StockfishConfig
from chessmaxx.training.dataset import write_training_examples
from chessmaxx.training.label import TrainingLabeler
from chessmaxx.training.split import split_game_ids, validate_game_isolation


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chessmaxx-data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build", help="sample PGN positions and label them with Stockfish"
    )
    build.add_argument("--pgn", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--count", type=_positive_int, required=True)
    build.add_argument("--seed", type=int, default=2026)
    build.add_argument("--minimum-ply", type=int, default=8)
    build.add_argument("--max-per-game", type=_positive_int, default=4)
    build.add_argument("--validation-fraction", type=_fraction, default=0.1)
    build.add_argument("--test-fraction", type=_fraction, default=0.0)
    build.add_argument("--minimum-train", type=_nonnegative_int, default=0)
    build.add_argument("--minimum-validation", type=_nonnegative_int, default=0)
    build.add_argument("--minimum-test", type=_nonnegative_int, default=0)
    build.add_argument("--stockfish", default="stockfish")
    build.add_argument(
        "--cache", type=Path, default=Path("artifacts/training-stockfish-cache.json")
    )
    build.add_argument("--journal", type=Path)
    build.add_argument("--nodes", type=_positive_int, default=50_000)
    build.add_argument("--multipv", type=_positive_int, default=3)
    build.add_argument("--threads", type=_positive_int, default=1)
    build.add_argument("--hash-mb", type=_positive_int, default=64)
    return parser


def run_build(args: argparse.Namespace) -> int:
    if args.minimum_ply < 0:
        raise ValueError("minimum_ply must be non-negative")
    if args.validation_fraction + args.test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than 1")
    positions = sample_pgn_positions(
        args.pgn,
        args.count,
        seed=args.seed,
        minimum_ply=args.minimum_ply,
        max_per_game=args.max_per_game,
    )
    if len(positions) != args.count:
        raise ValueError(
            f"requested {args.count} positions but the PGN supplied only "
            f"{len(positions)} eligible positions"
        )
    assignments = split_game_ids(
        (position.game_id or "" for position in positions),
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    position_split_counts = Counter(
        assignments[position.game_id or ""] for position in positions
    )
    required = {
        "train": args.minimum_train,
        "validation": args.minimum_validation,
        "test": args.minimum_test,
    }
    insufficient = {
        split: (position_split_counts.get(split, 0), minimum)
        for split, minimum in required.items()
        if position_split_counts.get(split, 0) < minimum
    }
    if insufficient:
        details = ", ".join(
            f"{split}={actual} (minimum {minimum})"
            for split, (actual, minimum) in insufficient.items()
        )
        raise ValueError(f"sampled split is too small: {details}")
    engine_config = StockfishConfig(
        nodes=args.nodes,
        multipv=args.multipv,
        threads=args.threads,
        hash_mb=args.hash_mb,
    )
    journal = args.journal or args.output.with_suffix(
        args.output.suffix + ".labels.progress.jsonl"
    )
    with StockfishAnalyzer.open(
        args.stockfish, config=engine_config, cache_path=args.cache
    ) as analyzer:
        engine_id = dict(analyzer.engine_id)
        examples = TrainingLabeler(analyzer, assignments).run(
            positions, journal_path=journal
        )
    validate_game_isolation(examples)
    write_training_examples(args.output, examples)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_pgn": str(args.pgn),
        "source_pgn_sha256": _sha256(args.pgn),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "count_requested": args.count,
        "count_written": len(examples),
        "seed": args.seed,
        "minimum_ply": args.minimum_ply,
        "max_per_game": args.max_per_game,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "minimum_split_counts": required,
        "split_counts": dict(Counter(example.split for example in examples)),
        "game_counts": dict(Counter(assignments.values())),
        "stockfish": asdict(engine_config),
        "engine": engine_id,
        "journal": str(journal),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        return run_build(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
