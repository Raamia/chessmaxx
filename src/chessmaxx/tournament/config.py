"""Typed TOML configuration and opening files for Elo tournaments."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from chessmaxx.tournament.schedule import OpeningPosition


@dataclass(frozen=True, slots=True)
class OpponentProfile:
    player_id: str
    kind: str
    rating: float | None = None
    move_time_ms: int = 100
    uci_elo: int | None = None
    skill_level: int | None = None

    def __post_init__(self) -> None:
        if not self.player_id.strip():
            raise ValueError("opponent player_id must not be empty")
        if self.kind not in {"random", "material", "stockfish"}:
            raise ValueError("opponent kind must be random, material, or stockfish")
        if self.rating is not None and self.kind != "stockfish":
            raise ValueError("only calibrated Stockfish opponents may declare Elo")
        if self.rating is not None and not 0 < self.rating < 5000:
            raise ValueError("opponent rating must be between zero and 5000")
        if self.move_time_ms <= 0:
            raise ValueError("move_time_ms must be positive")


@dataclass(frozen=True, slots=True)
class EloProfile:
    name: str
    model_id: str
    revision: str
    model_player_id: str
    selection: str
    games_per_opponent: int
    max_plies: int
    batch_size: int
    candidate_batch_size: int
    seed: int
    opponents: tuple[OpponentProfile, ...]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.model_id.strip():
            raise ValueError("Elo profile name and model_id must not be empty")
        if self.selection not in {"greedy", "legal-rerank"}:
            raise ValueError("selection must be greedy or legal-rerank")
        if self.games_per_opponent <= 0 or self.games_per_opponent % 2:
            raise ValueError("games_per_opponent must be positive and even")
        if min(self.max_plies, self.batch_size, self.candidate_batch_size) <= 0:
            raise ValueError("Elo batch and game limits must be positive")
        ids = [opponent.player_id for opponent in self.opponents]
        if not ids or len(ids) != len(set(ids)) or self.model_player_id in ids:
            raise ValueError("Elo profile requires unique opponent IDs")


def load_elo_profile(path: str | Path) -> EloProfile:
    source = Path(path)
    try:
        value = tomllib.loads(source.read_text(encoding="utf-8"))["elo"]
        opponents = tuple(
            OpponentProfile(**opponent) for opponent in value.pop("opponents")
        )
        return EloProfile(opponents=opponents, **value)
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid Elo profile {source}: {exc}") from exc


def load_openings(path: str | Path) -> tuple[OpeningPosition, ...]:
    source = Path(path)
    openings: list[OpeningPosition] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value: dict[str, Any] = json.loads(line)
                opening_id = str(value["opening_id"])
                moves = [str(move) for move in value.get("moves_uci", [])]
                board = chess.Board()
                for move in moves:
                    board.push_uci(move)
                opening = OpeningPosition(opening_id, board.fen())
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{source}:{line_number}: invalid opening: {exc}"
                ) from exc
            if opening.opening_id in seen:
                raise ValueError(f"{source}:{line_number}: duplicate opening_id")
            seen.add(opening.opening_id)
            openings.append(opening)
    if not openings:
        raise ValueError("opening file must not be empty")
    return tuple(openings)
