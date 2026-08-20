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


class RetryGenerator(FirstLegalGenerator):
    def __init__(self):
        super().__init__()
        self.prompt_batches = []

    def generate_prompts(self, prompts):
        self.prompt_batches.append(list(prompts))
        self.positions += len(prompts)
        move = "e2e5" if len(self.prompt_batches) == 1 else "e2e4"
        return [GeneratedMove(move, latency_ms=0.1) for _ in prompts]


class HistoryGenerator(FirstLegalGenerator):
    def __init__(self):
        super().__init__()
        self.prompts = []

    def generate_prompts(self, prompts):
        self.prompts.extend(prompts)
        self.positions += len(prompts)
        moves = ("e2e4", "a2a3")
        return [GeneratedMove(moves[len(self.prompts) - 1], latency_ms=0.1)]


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


def test_runner_batches_retry_feedback_across_games():
    model = RetryGenerator()
    opponent = FirstLegalGenerator()
    runner = TournamentRunner(
        {"model": model, "opponent": opponent},
        batch_size=3,
        max_plies=1,
        assisted_player_id="model",
        max_attempts=3,
        include_legal_moves=True,
    )

    results = runner.run(schedules(3))

    assert len(model.prompt_batches) == 2
    assert all("previous" not in prompt.lower() for prompt in model.prompt_batches[0])
    assert all("e2e5" in prompt for prompt in model.prompt_batches[1])
    assert all("Legal moves:" in prompt for prompt in model.prompt_batches[1])
    assert all(len(result.moves[0].attempts) == 2 for result in results)
    assert all(result.termination == "max_plies" for result in results)


def test_runner_can_add_history_without_enabling_retries():
    model = HistoryGenerator()
    opponent = FirstLegalGenerator()
    runner = TournamentRunner(
        {"model": model, "opponent": opponent},
        batch_size=1,
        max_plies=3,
        assisted_player_id="model",
        include_move_history=True,
    )

    result = runner.run(schedules(1))[0]

    assert "Moves played" not in model.prompts[0]
    assert "Moves played since the frozen opening: 1. e4 a5" in model.prompts[1]
    assert result.termination == "max_plies"
