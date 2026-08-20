"""Compare tournament reports only after proving their schedules match."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


def _schedule_fingerprint(report: dict[str, Any]) -> str:
    schedule = [
        {
            "game_id": game["game_id"],
            "opening_id": game["opening_id"],
            "initial_fen": game["initial_fen"],
            "white_id": game["white_id"],
            "black_id": game["black_id"],
        }
        for game in report["games"]
    ]
    encoded = json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ladder_fingerprint(report: dict[str, Any]) -> str:
    model_id = report["settings"]["profile"]["model_player_id"]
    opponents = {
        player_id: metadata
        for player_id, metadata in report["players"].items()
        if player_id != model_id
    }
    encoded = json.dumps(opponents, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _entry(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    settings = report["settings"]
    summary = report["summary"]
    model_id = settings["profile"]["model_player_id"]
    return {
        "report": str(path.resolve()),
        "model_source": settings["model_source"],
        "adapter_sha256": settings.get("adapter_sha256"),
        "selection": settings["selection"],
        "context": settings.get("context", "fen"),
        "max_attempts": settings.get("max_attempts", 1),
        "model": report["players"][model_id],
        "games": summary["games"],
        "wins": summary["wins"],
        "draws": summary["draws"],
        "losses": summary["losses"],
        "score_rate": summary["score_rate"],
        "first_attempt_legal_rate": summary.get(
            "model_first_attempt_legal_rate"
        ),
        "eventual_legal_rate": summary.get("model_eventual_legal_move_rate"),
        "selected_move_legal_rate": summary["model_selected_move_legal_rate"],
        "mean_attempts_per_move": summary.get("model_mean_attempts_per_move"),
        "illegal_forfeits": summary["model_illegal_forfeits"],
        "calibrated_elo": summary["calibrated_elo"],
    }


def compare_reports(paths: Sequence[str | Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("comparison requires at least two tournament reports")
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for source in paths:
        path = Path(source)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["schema_version"] != 1:
                raise ValueError("unsupported report schema")
            _schedule_fingerprint(report)
            _ladder_fingerprint(report)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid tournament report {path}: {exc}") from exc
        loaded.append((path, report))
    fingerprints = {_schedule_fingerprint(report) for _, report in loaded}
    if len(fingerprints) != 1:
        raise ValueError("tournament reports do not share the same game schedule")
    ladder_fingerprints = {_ladder_fingerprint(report) for _, report in loaded}
    if len(ladder_fingerprints) != 1:
        raise ValueError("tournament reports do not share the same opponent ladder")
    entries = [_entry(path, report) for path, report in loaded]
    conditions: set[tuple[str, str, str, int]] = set()
    for entry in entries:
        condition = (
            entry["model_source"],
            entry["selection"],
            entry["context"],
            entry["max_attempts"],
        )
        if condition in conditions:
            raise ValueError(f"duplicate comparison condition: {condition}")
        conditions.add(condition)
    deltas: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for entry in entries:
        key = (entry["selection"], entry["context"], entry["max_attempts"])
        grouped.setdefault(key, {})[entry["model_source"]] = entry
    for (selection, context, max_attempts), sources in sorted(grouped.items()):
        if {"base", "adapter"} <= sources.keys():
            base = sources["base"]
            adapter = sources["adapter"]
            deltas.append(
                {
                    "selection": selection,
                    "context": context,
                    "max_attempts": max_attempts,
                    "score_rate_delta": adapter["score_rate"] - base["score_rate"],
                    "first_attempt_legal_rate_delta": _optional_delta(
                        adapter["first_attempt_legal_rate"],
                        base["first_attempt_legal_rate"],
                    ),
                    "eventual_legal_rate_delta": _optional_delta(
                        adapter["eventual_legal_rate"],
                        base["eventual_legal_rate"],
                    ),
                }
            )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "schedule_sha256": fingerprints.pop(),
        "ladder_sha256": ladder_fingerprints.pop(),
        "conditions": entries,
        "adapter_minus_base": deltas,
    }


def _optional_delta(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chessmaxx-elo-compare")
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comparison = compare_reports(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
