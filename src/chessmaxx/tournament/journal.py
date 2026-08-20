"""Append-only completed-game journal for interruption-safe tournaments."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from chessmaxx.tournament.schema import GameResult, ScheduledGame


class TournamentJournalError(ValueError):
    pass


def tournament_run_key(
    schedules: Sequence[ScheduledGame],
    *,
    players: dict[str, dict[str, Any]],
    settings: dict[str, Any],
) -> str:
    payload = {
        "schedules": [
            {
                "game_id": game.game_id,
                "opening_id": game.opening_id,
                "initial_fen": game.initial_fen,
                "white_id": game.white_id,
                "black_id": game.black_id,
                "seed": game.seed,
            }
            for game in schedules
        ],
        "players": players,
        "settings": settings,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


class TournamentJournal:
    def __init__(self, path: str | Path, run_key: str) -> None:
        self.path = Path(path)
        self.run_key = run_key

    def load_or_create(self) -> dict[str, GameResult]:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            manifest = {
                "type": "manifest",
                "run_key": self.run_key,
                "created_at": datetime.now(UTC).isoformat(),
            }
            with self.path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return {}

        results: dict[str, GameResult] = {}
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TournamentJournalError(
                        f"{self.path}:{line_number}: corrupted JSON"
                    ) from exc
                if line_number == 1:
                    if value.get("type") != "manifest":
                        raise TournamentJournalError(
                            f"{self.path}: missing tournament manifest"
                        )
                    if value.get("run_key") != self.run_key:
                        raise TournamentJournalError(
                            f"{self.path}: games belong to a different tournament"
                        )
                    continue
                if value.get("type") != "game":
                    raise TournamentJournalError(
                        f"{self.path}:{line_number}: unknown journal record"
                    )
                result = GameResult.from_dict(value["value"])
                if result.game_id in results:
                    raise TournamentJournalError(
                        f"{self.path}:{line_number}: duplicate game result"
                    )
                results[result.game_id] = result
        if not self.path.read_text(encoding="utf-8").strip():
            raise TournamentJournalError(
                f"{self.path}: missing tournament manifest"
            )
        return results

    def append(self, result: GameResult) -> None:
        record = {"type": "game", "value": result.to_dict()}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
