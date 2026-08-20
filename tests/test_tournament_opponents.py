from types import SimpleNamespace

import chess

from chessmaxx.tournament.opponents import (
    DeterministicLegalMoveGenerator,
    StockfishMoveGenerator,
)


def test_random_opponent_is_stable_across_instances():
    first = DeterministicLegalMoveGenerator("random", kind="random", seed=7)
    second = DeterministicLegalMoveGenerator("random", kind="random", seed=7)

    assert first.generate_many([chess.STARTING_FEN])[0].raw_output == (
        second.generate_many([chess.STARTING_FEN])[0].raw_output
    )


def test_material_opponent_takes_a_hanging_queen():
    board = chess.Board("4k3/8/8/8/3q4/2P5/8/4K3 w - - 0 1")
    opponent = DeterministicLegalMoveGenerator("material", kind="material")

    response = opponent.generate_many([board.fen()])[0]

    assert response.raw_output == "c3d4"


def test_stockfish_opponent_uses_fixed_move_time_and_reports_identity():
    class Engine:
        id = {"name": "Fakefish", "author": "tests"}

        def __init__(self):
            self.limit = None

        def play(self, board, limit):
            self.limit = limit
            return SimpleNamespace(move=chess.Move.from_uci("e2e4"))

        def quit(self):
            pass

    engine = Engine()
    opponent = StockfishMoveGenerator(
        engine,  # type: ignore[arg-type]
        player_id="sf-1320",
        rating=1320,
        move_time_ms=50,
        settings={"UCI_Elo": 1320},
    )

    response = opponent.generate_many([chess.STARTING_FEN])[0]

    assert response.raw_output == "e2e4"
    assert engine.limit.time == 0.05
    assert opponent.metadata["rating"] == 1320
