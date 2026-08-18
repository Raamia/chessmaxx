"""Strict parsing and legality checks for model-generated chess moves."""

from __future__ import annotations

import re
from dataclasses import dataclass

import chess


UCI_MOVE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class MoveCheck:
    """The result of interpreting a model response as one UCI move."""

    raw_output: str
    candidate: str | None
    parsed_move: str | None
    is_legal: bool
    error: str | None = None


def first_token(raw_output: str) -> str | None:
    """Return the first whitespace-delimited token without searching later text."""

    tokens = raw_output.strip().split()
    return tokens[0] if tokens else None


def check_generated_move(fen: str, raw_output: str) -> MoveCheck:
    """Parse the first generated token as UCI and check it against the board."""

    board = chess.Board(fen)
    candidate = first_token(raw_output)
    if candidate is None:
        return MoveCheck(raw_output, None, None, False, "empty_output")

    normalized = candidate.lower()
    if not UCI_MOVE.fullmatch(normalized):
        return MoveCheck(raw_output, candidate, None, False, "invalid_uci")

    try:
        move = chess.Move.from_uci(normalized)
    except ValueError:
        return MoveCheck(raw_output, candidate, None, False, "invalid_uci")

    if move not in board.legal_moves:
        return MoveCheck(raw_output, candidate, move.uci(), False, "illegal_move")
    return MoveCheck(raw_output, candidate, move.uci(), True)

