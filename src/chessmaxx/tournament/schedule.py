"""Deterministic paired-color tournament scheduling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import chess

from chessmaxx.tournament.schema import ScheduledGame


@dataclass(frozen=True, slots=True)
class OpeningPosition:
    opening_id: str
    fen: str

    def __post_init__(self) -> None:
        if not self.opening_id.strip():
            raise ValueError("opening_id must not be empty")
        board = chess.Board(self.fen)
        if not board.is_valid() or board.is_game_over():
            raise ValueError("opening must be valid and non-terminal")


def paired_schedule(
    *,
    model_id: str,
    opponent_ids: Sequence[str],
    openings: Sequence[OpeningPosition],
    games_per_opponent: int,
    seed: int,
) -> tuple[ScheduledGame, ...]:
    """Use each selected opening twice with model colors reversed."""

    if not model_id.strip() or not opponent_ids:
        raise ValueError("schedule requires a model and at least one opponent")
    if games_per_opponent <= 0 or games_per_opponent % 2:
        raise ValueError("games_per_opponent must be positive and even")
    if not openings:
        raise ValueError("schedule requires at least one opening")
    if model_id in opponent_ids or len(set(opponent_ids)) != len(opponent_ids):
        raise ValueError("opponent IDs must be unique and different from the model")
    pair_count = games_per_opponent // 2
    schedules: list[ScheduledGame] = []
    for opponent_id in opponent_ids:
        for pair_index in range(pair_count):
            opening = openings[pair_index % len(openings)]
            pair_key = f"{seed}\0{opponent_id}\0{pair_index}\0{opening.opening_id}"
            pair_seed = int.from_bytes(
                hashlib.sha256(pair_key.encode()).digest()[:8], "big"
            )
            for model_is_white in (True, False):
                color = "white" if model_is_white else "black"
                schedules.append(
                    ScheduledGame(
                        game_id=f"{opponent_id}-pair-{pair_index:04d}-model-{color}",
                        opening_id=opening.opening_id,
                        initial_fen=opening.fen,
                        white_id=model_id if model_is_white else opponent_id,
                        black_id=opponent_id if model_is_white else model_id,
                        seed=pair_seed,
                    )
                )
    return tuple(schedules)
