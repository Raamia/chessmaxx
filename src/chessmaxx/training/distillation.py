"""Multi-PV policy targets derived from Stockfish centipawn utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from chessmaxx.evaluation.schema import TeacherMove


@dataclass(frozen=True, slots=True)
class PolicyTarget:
    move: str
    score_cp: int
    probability: float


def centipawn_policy(
    teacher_moves: Sequence[TeacherMove],
    *,
    temperature_cp: float,
    max_candidates: int,
) -> tuple[PolicyTarget, ...]:
    """Turn ranked engine utilities into a normalized candidate policy."""

    if temperature_cp <= 0:
        raise ValueError("temperature_cp must be positive")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    candidates = list(teacher_moves[:max_candidates])
    if not candidates:
        raise ValueError("teacher policy requires at least one move")
    maximum = max(move.score_cp for move in candidates)
    weights = [
        math.exp((move.score_cp - maximum) / temperature_cp)
        for move in candidates
    ]
    total = sum(weights)
    return tuple(
        PolicyTarget(
            move=move.move,
            score_cp=move.score_cp,
            probability=weight / total,
        )
        for move, weight in zip(candidates, weights, strict=True)
    )


def policy_entropy(policy: Sequence[PolicyTarget]) -> float:
    if not policy:
        raise ValueError("cannot measure an empty policy")
    return -sum(
        target.probability * math.log(target.probability)
        for target in policy
        if target.probability > 0
    )
