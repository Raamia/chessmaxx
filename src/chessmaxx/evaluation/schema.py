"""Serializable records shared by evaluation components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chess


@dataclass(frozen=True, slots=True)
class TeacherMove:
    """A Stockfish candidate and its score from the side to move."""

    move: str
    score_cp: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherMove":
        return cls(move=str(value["move"]), score_cp=int(value["score_cp"]))

    def to_dict(self) -> dict[str, Any]:
        return {"move": self.move, "score_cp": self.score_cp}


@dataclass(frozen=True, slots=True)
class EvaluationPosition:
    """One immutable position in a frozen evaluation set."""

    position_id: str
    fen: str
    game_id: str | None = None
    ply: int | None = None
    phase: str | None = None
    teacher_moves: tuple[TeacherMove, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.position_id.strip():
            raise ValueError("position_id must not be empty")
        if not chess.Board(self.fen).is_valid():
            raise ValueError(f"position {self.position_id!r} has an invalid board")
        if self.ply is not None and self.ply < 0:
            raise ValueError("ply must be non-negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationPosition":
        return cls(
            position_id=str(value["position_id"]),
            fen=str(value["fen"]),
            game_id=value.get("game_id"),
            ply=value.get("ply"),
            phase=value.get("phase"),
            teacher_moves=tuple(
                TeacherMove.from_dict(move) for move in value.get("teacher_moves", [])
            ),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "position_id": self.position_id,
            "fen": self.fen,
            "teacher_moves": [move.to_dict() for move in self.teacher_moves],
        }
        for key in ("game_id", "ply", "phase"):
            item = getattr(self, key)
            if item is not None:
                value[key] = item
        if self.metadata:
            value["metadata"] = self.metadata
        return value

