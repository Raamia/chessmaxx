"""Auditable records shared by tournament scheduling, play, and reports."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import chess


RESULTS = {"1-0", "0-1", "1/2-1/2"}
TERMINATIONS = {
    "checkmate",
    "stalemate",
    "insufficient_material",
    "seventyfive_moves",
    "fivefold_repetition",
    "max_plies",
    "illegal_move",
}


@dataclass(frozen=True, slots=True)
class PlayerSpec:
    player_id: str
    kind: str
    rating: float | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.player_id.strip():
            raise ValueError("player_id must not be empty")
        if self.kind not in {"model", "random", "material", "stockfish"}:
            raise ValueError("unsupported tournament player kind")
        if self.rating is not None and not 0 < self.rating < 5000:
            raise ValueError("player rating must be between 0 and 5000")


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    game_id: str
    opening_id: str
    initial_fen: str
    white_id: str
    black_id: str
    seed: int

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.game_id,
                self.opening_id,
                self.white_id,
                self.black_id,
            )
        ):
            raise ValueError("scheduled game identifiers must not be empty")
        board = chess.Board(self.initial_fen)
        if not board.is_valid() or board.is_game_over():
            raise ValueError("scheduled game requires a valid non-terminal FEN")
        if self.white_id == self.black_id:
            raise ValueError("scheduled game players must be different")


@dataclass(frozen=True, slots=True)
class MoveAttempt:
    attempt: int
    raw_output: str
    move_uci: str | None
    legal: bool
    error: str | None
    latency_ms: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.attempt <= 0 or self.latency_ms < 0:
            raise ValueError("move attempt number must be positive and latency non-negative")
        for name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if value is not None and value < 0:
                raise ValueError(f"move attempt {name} must be non-negative")
        if self.legal and (self.move_uci is None or self.error is not None):
            raise ValueError("legal move attempt requires UCI and no error")
        if not self.legal and self.error is None:
            raise ValueError("illegal move attempt requires an error")


@dataclass(frozen=True, slots=True)
class MoveRecord:
    ply: int
    fen_before: str
    player_id: str
    raw_output: str
    move_uci: str | None
    legal: bool
    latency_ms: float
    attempts: tuple[MoveAttempt, ...] = ()

    def __post_init__(self) -> None:
        if self.ply < 0 or self.latency_ms < 0:
            raise ValueError("move ply and latency must be non-negative")
        if not self.player_id.strip():
            raise ValueError("move player_id must not be empty")
        board = chess.Board(self.fen_before)
        if not board.is_valid():
            raise ValueError("move record contains an invalid FEN")
        if self.legal:
            if self.move_uci is None:
                raise ValueError("legal move record requires move_uci")
            try:
                move = chess.Move.from_uci(self.move_uci)
            except ValueError as exc:
                raise ValueError("legal move record contains invalid UCI") from exc
            if move not in board.legal_moves:
                raise ValueError("move marked legal is illegal for its FEN")
        if self.attempts:
            if any(
                attempt.attempt != index
                for index, attempt in enumerate(self.attempts, start=1)
            ):
                raise ValueError("move attempts must use contiguous one-based numbers")
            if any(attempt.legal for attempt in self.attempts[:-1]):
                raise ValueError("a legal attempt must finish the move")
            final = self.attempts[-1]
            if (final.raw_output, final.move_uci, final.legal) != (
                self.raw_output,
                self.move_uci,
                self.legal,
            ):
                raise ValueError("final attempt must match the move record")
            if not math.isclose(
                self.latency_ms,
                sum(attempt.latency_ms for attempt in self.attempts),
            ):
                raise ValueError("move latency must equal total attempt latency")
            for attempt in self.attempts:
                if attempt.legal:
                    try:
                        move = chess.Move.from_uci(attempt.move_uci or "")
                    except ValueError as exc:
                        raise ValueError(
                            "legal move attempt contains invalid UCI"
                        ) from exc
                    if move not in board.legal_moves:
                        raise ValueError("legal move attempt is illegal for its FEN")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MoveRecord":
        restored = dict(value)
        restored["attempts"] = tuple(
            MoveAttempt(**attempt) for attempt in value.get("attempts", ())
        )
        return cls(**restored)


@dataclass(frozen=True, slots=True)
class GameResult:
    game_id: str
    opening_id: str
    initial_fen: str
    white_id: str
    black_id: str
    result: str
    termination: str
    final_fen: str
    moves: tuple[MoveRecord, ...]

    def __post_init__(self) -> None:
        if self.result not in RESULTS:
            raise ValueError("unsupported chess result")
        if self.termination not in TERMINATIONS:
            raise ValueError("unsupported game termination")
        if not chess.Board(self.final_fen).is_valid():
            raise ValueError("game result contains an invalid final FEN")
        if any(move.ply != index for index, move in enumerate(self.moves)):
            raise ValueError("game moves must use contiguous zero-based plies")

    def score_for(self, player_id: str) -> float:
        if player_id not in {self.white_id, self.black_id}:
            raise ValueError("player did not participate in this game")
        if self.result == "1/2-1/2":
            return 0.5
        white_won = self.result == "1-0"
        return float(
            (player_id == self.white_id and white_won)
            or (player_id == self.black_id and not white_won)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GameResult":
        return cls(
            game_id=str(value["game_id"]),
            opening_id=str(value["opening_id"]),
            initial_fen=str(value["initial_fen"]),
            white_id=str(value["white_id"]),
            black_id=str(value["black_id"]),
            result=str(value["result"]),
            termination=str(value["termination"]),
            final_fen=str(value["final_fen"]),
            moves=tuple(MoveRecord.from_dict(move) for move in value["moves"]),
        )
