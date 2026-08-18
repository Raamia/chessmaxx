import chess
import pytest

from chessmaxx.evaluation.moves import check_generated_move


@pytest.mark.parametrize("output", ["e2e4", "e2e4\n", "  E2E4 explanation"])
def test_accepts_legal_first_uci_token(output):
    result = check_generated_move(chess.STARTING_FEN, output)

    assert result.parsed_move == "e2e4"
    assert result.is_legal is True
    assert result.error is None


@pytest.mark.parametrize(
    ("output", "error"),
    [
        ("", "empty_output"),
        ("Move: e2e4", "invalid_uci"),
        ("e9e4", "invalid_uci"),
        ("e2e5", "illegal_move"),
    ],
)
def test_rejects_unusable_first_moves(output, error):
    result = check_generated_move(chess.STARTING_FEN, output)

    assert result.is_legal is False
    assert result.error == error


def test_accepts_promotion_suffix():
    fen = "7k/P7/8/8/8/8/8/6K1 w - - 0 1"

    result = check_generated_move(fen, "a7a8q")

    assert result.parsed_move == "a7a8q"
    assert result.is_legal is True
