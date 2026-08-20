import chess
import pytest

from chessmaxx.tournament.schema import (
    GameResult,
    MoveRecord,
    PlayerSpec,
    ScheduledGame,
)


def test_game_result_round_trips_and_scores_both_colors():
    move = MoveRecord(
        ply=0,
        fen_before=chess.STARTING_FEN,
        player_id="model",
        raw_output="e2e4",
        move_uci="e2e4",
        legal=True,
        latency_ms=12.5,
    )
    board = chess.Board()
    board.push_uci("e2e4")
    result = GameResult(
        game_id="game-1",
        opening_id="start",
        initial_fen=chess.STARTING_FEN,
        white_id="model",
        black_id="opponent",
        result="1-0",
        termination="illegal_move",
        final_fen=board.fen(),
        moves=(move,),
    )

    restored = GameResult.from_dict(result.to_dict())

    assert restored == result
    assert restored.score_for("model") == 1.0
    assert restored.score_for("opponent") == 0.0


def test_schedule_rejects_terminal_opening():
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    board.push_san("g4")
    board.push_san("Qh4#")

    with pytest.raises(ValueError, match="non-terminal"):
        ScheduledGame(
            game_id="bad",
            opening_id="mate",
            initial_fen=board.fen(),
            white_id="a",
            black_id="b",
            seed=1,
        )


def test_player_spec_distinguishes_calibrated_and_uncalibrated_opponents():
    assert PlayerSpec("random", "random").rating is None
    assert PlayerSpec("anchor", "stockfish", rating=1320).rating == 1320

    with pytest.raises(ValueError, match="rating"):
        PlayerSpec("bad", "stockfish", rating=-10)
