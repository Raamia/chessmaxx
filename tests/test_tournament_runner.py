import chess

from chessmaxx.evaluation.model import GeneratedMove
from chessmaxx.tournament.runner import TournamentRunner
from chessmaxx.tournament.schema import ScheduledGame


class FirstLegalGenerator:
    metadata = {"kind": "fixture"}

    def __init__(self):
        self.batch_sizes = []
        self.reset_telemetry()

    def reset_telemetry(self):
        self.positions = 0

    @property
    def telemetry(self):
        return {"positions_generated": self.positions}

    def generate_many(self, fens):
        self.batch_sizes.append(len(fens))
        self.positions += len(fens)
        return [
            GeneratedMove(
                sorted(move.uci() for move in chess.Board(fen).legal_moves)[0],
                latency_ms=0.1,
            )
            for fen in fens
        ]


def schedules(count):
    return [
        ScheduledGame(
            game_id=f"game-{index}",
            opening_id="start",
            initial_fen=chess.STARTING_FEN,
            white_id="model",
            black_id="opponent",
            seed=index,
        )
        for index in range(count)
    ]


def test_runner_batches_same_player_turns_across_games():
    model = FirstLegalGenerator()
    opponent = FirstLegalGenerator()
    completed = []
    runner = TournamentRunner(
        {"model": model, "opponent": opponent},
        batch_size=3,
        max_plies=2,
        on_result=completed.append,
    )

    results = runner.run(schedules(5))

    assert [result.game_id for result in results] == [f"game-{i}" for i in range(5)]
    assert all(result.result == "1/2-1/2" for result in results)
    assert model.batch_sizes == [3, 2]
    assert opponent.batch_sizes == [3, 2]
    assert len(completed) == 5
    assert runner.telemetry["model"]["positions_generated"] == 5


def test_runner_rejects_missing_player_generator():
    runner = TournamentRunner({"model": FirstLegalGenerator()})

    try:
        runner.run(schedules(1))
    except ValueError as exc:
        assert "opponent" in str(exc)
    else:
        raise AssertionError("expected missing player to fail")
