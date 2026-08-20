import chess

from chessmaxx.evaluation.model import build_prompt
from chessmaxx.tournament.prompts import build_retry_prompt, san_history
from chessmaxx.tournament.schema import MoveAttempt


def illegal_attempt() -> MoveAttempt:
    return MoveAttempt(
        attempt=1,
        raw_output="e2e5",
        move_uci="e2e5",
        legal=False,
        error="illegal_move",
        latency_ms=1.0,
    )


def test_first_retry_prompt_matches_the_canonical_fen_prompt():
    prompt = build_retry_prompt(chess.STARTING_FEN, ())

    assert prompt == build_prompt(chess.STARTING_FEN)


def test_retry_prompt_explains_failure_and_can_reveal_legal_moves():
    prompt = build_retry_prompt(
        chess.STARTING_FEN,
        (illegal_attempt(),),
        include_legal_moves=True,
    )

    assert "e2e5" in prompt
    assert "illegal in the current position" in prompt
    assert "Try again from the unchanged position" in prompt
    assert "Legal moves:" in prompt
    assert "e2e4" in prompt


def test_prompt_can_include_actual_game_history():
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    history = san_history(board)

    prompt = build_retry_prompt(board.fen(), (), move_history=history)

    assert history == "1. e4 e5"
    assert "Moves played since the frozen opening: 1. e4 e5" in prompt
    assert board.fen() in prompt
