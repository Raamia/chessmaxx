"""Prompts for auditable retry-based chess assistance."""

from __future__ import annotations

import chess

from chessmaxx.evaluation.model import build_prompt
from chessmaxx.tournament.schema import MoveAttempt


_ERROR_MESSAGES = {
    "empty_output": "the response was empty",
    "invalid_uci": "the response was not exactly one UCI move",
    "illegal_move": "the move is illegal in the current position",
}


def build_retry_prompt(
    fen: str,
    attempts: tuple[MoveAttempt, ...],
    *,
    include_legal_moves: bool = False,
    move_history: str | None = None,
) -> str:
    """Repeat the unchanged board and explain rejected attempts."""

    if not attempts and not include_legal_moves and not move_history:
        return build_prompt(fen)
    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over():
        raise ValueError("retry prompt requires a valid non-terminal FEN")
    lines = [
        "Choose the best chess move for the position below. "
        "Respond with exactly one move in UCI notation and no explanation.",
    ]
    if attempts:
        lines.append("Your previous attempt(s) were rejected:")
        for attempt in attempts:
            output = " ".join(attempt.raw_output.split()) or "<empty>"
            reason = _ERROR_MESSAGES.get(
                attempt.error or "", attempt.error or "unknown error"
            )
            lines.append(f"{attempt.attempt}. {output!r}: {reason}.")
        lines.append("Try again from the unchanged position.")
    if include_legal_moves:
        legal_moves = " ".join(sorted(move.uci() for move in board.legal_moves))
        lines.append(f"Legal moves: {legal_moves}")
    if move_history:
        lines.append(f"Moves played since the frozen opening: {move_history}")
    lines.extend((f"FEN: {fen}", "Move:"))
    return "\n".join(lines)


def san_history(board: chess.Board) -> str | None:
    """Render moves played after the scheduled opening without changing the board."""

    if not board.move_stack:
        return None
    return board.root().variation_san(board.move_stack)
