"""Versioned records used by the Stockfish-supervised training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess

from chessmaxx.evaluation.schema import TeacherMove


TRAINING_SCHEMA_VERSION = 1
PROMPT_VERSION = "fen-uci-v1"


@dataclass(frozen=True, slots=True)
class TrainingExample:
    example_id: str
    game_id: str
    ply: int
    fen: str
    target_move: str
    teacher_moves: tuple[TeacherMove, ...]
    split: str
    source: str
    schema_version: int = TRAINING_SCHEMA_VERSION
    prompt_version: str = PROMPT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_SCHEMA_VERSION:
            raise ValueError(f"unsupported training schema {self.schema_version}")
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError(f"unsupported prompt version {self.prompt_version!r}")
        if not self.example_id.strip() or not self.game_id.strip():
            raise ValueError("example_id and game_id must not be empty")
        if self.ply < 0:
            raise ValueError("ply must be non-negative")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        board = chess.Board(self.fen)
        if not board.is_valid() or board.is_game_over():
            raise ValueError("training position must be valid and non-terminal")
        if not self.teacher_moves:
            raise ValueError("teacher_moves must not be empty")
        seen: set[str] = set()
        previous_score: int | None = None
        for teacher in self.teacher_moves:
            try:
                move = chess.Move.from_uci(teacher.move)
            except ValueError as exc:
                raise ValueError(f"invalid teacher move {teacher.move!r}") from exc
            if move not in board.legal_moves:
                raise ValueError(f"illegal teacher move {teacher.move!r}")
            if teacher.move in seen:
                raise ValueError(f"duplicate teacher move {teacher.move!r}")
            if previous_score is not None and teacher.score_cp > previous_score:
                raise ValueError("teacher moves must be ordered by descending score")
            seen.add(teacher.move)
            previous_score = teacher.score_cp
        if self.target_move != self.teacher_moves[0].move:
            raise ValueError("target_move must equal the highest-ranked teacher move")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingExample":
        return cls(
            example_id=str(value["example_id"]),
            game_id=str(value["game_id"]),
            ply=int(value["ply"]),
            fen=str(value["fen"]),
            target_move=str(value["target_move"]),
            teacher_moves=tuple(
                TeacherMove.from_dict(move) for move in value["teacher_moves"]
            ),
            split=str(value["split"]),
            source=str(value["source"]),
            schema_version=int(value.get("schema_version", TRAINING_SCHEMA_VERSION)),
            prompt_version=str(value.get("prompt_version", PROMPT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "prompt_version": self.prompt_version,
            "example_id": self.example_id,
            "game_id": self.game_id,
            "ply": self.ply,
            "fen": self.fen,
            "target_move": self.target_move,
            "teacher_moves": [move.to_dict() for move in self.teacher_moves],
            "split": self.split,
            "source": self.source,
        }

