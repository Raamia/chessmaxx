"""Multi-PV policy targets derived from Stockfish centipawn utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from chessmaxx.evaluation.schema import TeacherMove
from chessmaxx.training.schema import TrainingExample
from chessmaxx.training.tokenize import (
    EncodedExample,
    TokenizerLike,
    encode_prompt_target,
)


@dataclass(frozen=True, slots=True)
class PolicyTarget:
    move: str
    score_cp: int
    probability: float


@dataclass(frozen=True, slots=True)
class EncodedPolicyExample:
    example_id: str
    candidates: tuple[EncodedExample, ...]
    teacher_probabilities: tuple[float, ...]

    @property
    def candidate_tokens(self) -> int:
        return sum(len(candidate.input_ids) for candidate in self.candidates)


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


def encode_policy_example(
    example: TrainingExample,
    tokenizer: TokenizerLike,
    *,
    max_length: int,
    temperature_cp: float,
    max_candidates: int,
) -> EncodedPolicyExample:
    policy = centipawn_policy(
        example.teacher_moves,
        temperature_cp=temperature_cp,
        max_candidates=max_candidates,
    )
    candidates = tuple(
        encode_prompt_target(
            example_id=f"{example.example_id}:{target.move}",
            fen=example.fen,
            move_uci=target.move,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        for target in policy
    )
    return EncodedPolicyExample(
        example_id=example.example_id,
        candidates=candidates,
        teacher_probabilities=tuple(target.probability for target in policy),
    )


class PolicyTokenDataset:
    def __init__(
        self,
        examples: list[TrainingExample],
        tokenizer: TokenizerLike,
        *,
        max_length: int,
        temperature_cp: float,
        max_candidates: int,
    ) -> None:
        self.items = [
            encode_policy_example(
                example,
                tokenizer,
                max_length=max_length,
                temperature_cp=temperature_cp,
                max_candidates=max_candidates,
            )
            for example in examples
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {
            "example_id": item.example_id,
            "input_ids": [list(candidate.input_ids) for candidate in item.candidates],
            "attention_mask": [
                list(candidate.attention_mask) for candidate in item.candidates
            ],
            "labels": [list(candidate.labels) for candidate in item.candidates],
            "teacher_probabilities": list(item.teacher_probabilities),
        }
