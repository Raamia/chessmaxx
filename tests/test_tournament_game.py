import chess

from chessmaxx.evaluation.model import GeneratedMove
from chessmaxx.tournament.game import ActiveGame, result_to_pgn
from chessmaxx.tournament.schema import ScheduledGame


def schedule() -> ScheduledGame:
    return ScheduledGame(
        game_id="game-1",
        opening_id="start",
        initial_fen=chess.STARTING_FEN,
        white_id="white",
        black_id="black",
        seed=1,
    )


def generated(move: str) -> GeneratedMove:
    return GeneratedMove(move, latency_ms=1.0)


def test_game_adjudicates_checkmate_and_exports_pgn():
    game = ActiveGame(schedule())

    for move in ("f2f3", "e7e5", "g2g4"):
        assert game.apply(generated(move)) is None
    result = game.apply(generated("d8h4"))

    assert result is not None
    assert result.result == "0-1"
    assert result.termination == "checkmate"
    assert "1. f3 e5 2. g4 Qh4# 0-1" in result_to_pgn(result)


def test_illegal_model_move_is_an_immediate_forfeit():
    game = ActiveGame(schedule())

    result = game.apply(generated("e2e5"))

    assert result is not None
    assert result.result == "0-1"
    assert result.termination == "illegal_move"
    assert result.moves[0].legal is False
    assert result.moves[0].attempts[0].error == "illegal_move"
    pgn = result_to_pgn(result)
    assert '[IllegalPlayer "white"]' in pgn
    assert '[IllegalOutput "e2e5"]' in pgn


def test_maximum_ply_adjudication_is_a_draw():
    game = ActiveGame(schedule(), max_plies=2)

    assert game.apply(generated("g1f3")) is None
    result = game.apply(generated("g8f6"))

    assert result is not None
    assert result.result == "1/2-1/2"
    assert result.termination == "max_plies"


def test_assisted_player_can_correct_an_illegal_move_on_the_same_ply():
    game = ActiveGame(
        schedule(), assisted_player_id="white", max_attempts=3
    )

    assert game.apply(generated("e2e5")) is None
    assert game.board.fen() == chess.STARTING_FEN
    assert game.moves == []
    assert len(game.pending_attempts) == 1
    assert game.apply(generated("e2e4")) is None

    assert game.board.peek().uci() == "e2e4"
    assert len(game.moves) == 1
    assert [attempt.legal for attempt in game.moves[0].attempts] == [False, True]
    assert game.moves[0].latency_ms == 2.0


def test_assisted_player_forfeits_after_exhausting_retries():
    game = ActiveGame(
        schedule(), assisted_player_id="white", max_attempts=3
    )

    assert game.apply(generated("bad")) is None
    assert game.apply(generated("still-bad")) is None
    result = game.apply(generated("e2e5"))

    assert result is not None
    assert result.termination == "illegal_move"
    assert len(result.moves) == 1
    assert len(result.moves[0].attempts) == 3
