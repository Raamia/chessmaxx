"""Deterministic game-level splits that prevent neighboring-position leakage."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

from chessmaxx.training.schema import TrainingExample


def _split_count(total: int, fraction: float) -> int:
    if fraction == 0 or total == 0:
        return 0
    return max(1, math.floor(total * fraction))


def split_game_ids(
    game_ids: Iterable[str],
    *,
    seed: int = 0,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.0,
) -> dict[str, str]:
    """Assign every unique game to exactly one stable dataset split."""

    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than 1")
    unique = set(game_ids)
    if not unique:
        raise ValueError("at least one game_id is required")
    if any(not game_id.strip() for game_id in unique):
        raise ValueError("game_ids must not be empty")

    ranked = sorted(
        unique,
        key=lambda game_id: (
            hashlib.sha256(f"{seed}:{game_id}".encode()).digest(),
            game_id,
        ),
    )
    validation_count = _split_count(len(ranked), validation_fraction)
    test_count = _split_count(len(ranked), test_fraction)
    while validation_count + test_count >= len(ranked) and (
        validation_count or test_count
    ):
        if test_count >= validation_count and test_count:
            test_count -= 1
        elif validation_count:
            validation_count -= 1

    assignments: dict[str, str] = {}
    for index, game_id in enumerate(ranked):
        if index < test_count:
            split = "test"
        elif index < test_count + validation_count:
            split = "validation"
        else:
            split = "train"
        assignments[game_id] = split
    return assignments


def validate_game_isolation(examples: Iterable[TrainingExample]) -> None:
    """Reject a dataset when one source game occurs in multiple splits."""

    observed: dict[str, str] = {}
    for example in examples:
        previous = observed.setdefault(example.game_id, example.split)
        if previous != example.split:
            raise ValueError(
                f"game {example.game_id!r} appears in both {previous} and "
                f"{example.split}"
            )

