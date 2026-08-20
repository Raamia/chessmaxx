"""Command-line entry point for resumable Elo tournaments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from chessmaxx import __version__
from chessmaxx.evaluation.model import (
    HuggingFaceLegalMoveRanker,
    HuggingFaceMoveGenerator,
)
from chessmaxx.tournament.config import load_elo_profile, load_openings
from chessmaxx.tournament.game import result_to_pgn
from chessmaxx.tournament.journal import TournamentJournal, tournament_run_key
from chessmaxx.tournament.opponents import (
    DeterministicLegalMoveGenerator,
    StockfishMoveGenerator,
)
from chessmaxx.tournament.report import summarize_tournament
from chessmaxx.tournament.runner import TournamentRunner
from chessmaxx.tournament.schedule import paired_schedule


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("adapter directory contains no files")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chessmaxx-elo")
    parser.add_argument("--profile", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter-dir", type=Path)
    source.add_argument("--base-model-only", action="store_true")
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path)
    parser.add_argument("--device")
    parser.add_argument(
        "--selection",
        choices=("greedy", "retry", "retry-with-legal-list", "legal-rerank"),
    )
    parser.add_argument("--context", choices=("fen", "fen-pgn"))
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = load_elo_profile(args.profile)
    selection = args.selection or profile.selection
    context = args.context or profile.context
    retrying = selection in {"retry", "retry-with-legal-list"}
    configured_attempts = (
        args.max_attempts
        if args.max_attempts is not None
        else profile.max_attempts
    )
    max_attempts = configured_attempts if retrying else 1
    if max_attempts <= 0 or (retrying and max_attempts < 2):
        raise ValueError("retry modes require at least two move attempts")
    if args.max_attempts is not None and not retrying:
        raise ValueError("--max-attempts requires a retry selection mode")
    adapter_dir = args.adapter_dir.resolve() if args.adapter_dir else None
    openings = load_openings(args.openings)
    schedules = paired_schedule(
        model_id=profile.model_player_id,
        opponent_ids=[opponent.player_id for opponent in profile.opponents],
        openings=openings,
        games_per_opponent=profile.games_per_opponent,
        seed=profile.seed,
    )
    if selection == "legal-rerank":
        if adapter_dir is None:
            model = HuggingFaceLegalMoveRanker.from_pretrained(
                profile.model_id,
                revision=profile.revision,
                device=args.device,
                candidate_batch_size=profile.candidate_batch_size,
            )
        else:
            model = HuggingFaceLegalMoveRanker.from_adapter(
                adapter_dir,
                base_model_name=profile.model_id,
                revision=profile.revision,
                device=args.device,
                candidate_batch_size=profile.candidate_batch_size,
            )
    else:
        if adapter_dir is None:
            model = HuggingFaceMoveGenerator.from_pretrained(
                profile.model_id,
                revision=profile.revision,
                device=args.device,
                max_new_tokens=8,
            )
        else:
            model = HuggingFaceMoveGenerator.from_adapter(
                adapter_dir,
                base_model_name=profile.model_id,
                revision=profile.revision,
                device=args.device,
                max_new_tokens=8,
            )
    generators = {profile.model_player_id: model}
    closeable: list[StockfishMoveGenerator] = []
    try:
        for opponent in profile.opponents:
            if opponent.kind in {"random", "material"}:
                generators[opponent.player_id] = DeterministicLegalMoveGenerator(
                    opponent.player_id, kind=opponent.kind, seed=profile.seed
                )
            else:
                if args.stockfish is None:
                    raise ValueError("Stockfish opponent requires --stockfish")
                stockfish = StockfishMoveGenerator.open(
                    args.stockfish,
                    player_id=opponent.player_id,
                    rating=opponent.rating,
                    move_time_ms=opponent.move_time_ms,
                    threads=args.threads,
                    hash_mb=args.hash_mb,
                    uci_elo=opponent.uci_elo,
                    skill_level=opponent.skill_level,
                )
                generators[opponent.player_id] = stockfish
                closeable.append(stockfish)
        player_metadata = {
            player_id: dict(generator.metadata)
            for player_id, generator in generators.items()
        }
        settings = {
            "profile": asdict(profile),
            "profile_sha256": _sha256(args.profile),
            "openings_sha256": _sha256(args.openings),
            "model_source": "base" if adapter_dir is None else "adapter",
            "adapter_sha256": (
                _tree_sha256(adapter_dir) if adapter_dir is not None else None
            ),
            "selection": selection,
            "context": context,
            "max_attempts": max_attempts,
            "batch_size": profile.batch_size,
            "max_plies": profile.max_plies,
        }
        run_key = tournament_run_key(
            schedules, players=player_metadata, settings=settings
        )
        journal = TournamentJournal(args.journal, run_key)
        restored = journal.load_or_create()
        expected = {schedule.game_id: schedule for schedule in schedules}
        for game_id, result in restored.items():
            schedule = expected.get(game_id)
            if schedule is None or (
                result.opening_id,
                result.initial_fen,
                result.white_id,
                result.black_id,
            ) != (
                schedule.opening_id,
                schedule.initial_fen,
                schedule.white_id,
                schedule.black_id,
            ):
                raise ValueError(f"journal game {game_id!r} does not match schedule")
        pending = [schedule for schedule in schedules if schedule.game_id not in restored]
        runner = TournamentRunner(
            generators,
            batch_size=profile.batch_size,
            max_plies=profile.max_plies,
            assisted_player_id=(
                profile.model_player_id
                if retrying or context == "fen-pgn"
                else None
            ),
            max_attempts=max_attempts,
            include_legal_moves=selection == "retry-with-legal-list",
            include_move_history=context == "fen-pgn",
            on_result=journal.append,
        )
        generated = runner.run(pending)
        results_by_id = {**restored, **{result.game_id: result for result in generated}}
        results = tuple(results_by_id[schedule.game_id] for schedule in schedules)
        opponent_ratings = {
            opponent.player_id: opponent.rating
            for opponent in profile.opponents
            if opponent.rating is not None
        }
        report = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "chessmaxx_version": __version__,
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "run_key": run_key,
            "settings": settings,
            "players": player_metadata,
            "games_restored": len(restored),
            "telemetry": runner.telemetry,
            "summary": summarize_tournament(
                results,
                model_id=profile.model_player_id,
                opponent_ratings=opponent_ratings,
                selection=selection,
            ),
            "games": [result.to_dict() for result in results],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.pgn.parent.mkdir(parents=True, exist_ok=True)
        args.pgn.write_text(
            "\n".join(result_to_pgn(result).rstrip() for result in results) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        return 0
    finally:
        for generator in closeable:
            generator.close()


if __name__ == "__main__":
    raise SystemExit(main())
