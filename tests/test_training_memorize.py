import chess

from chessmaxx.evaluation.model import GeneratedMove
from chessmaxx.evaluation.schema import TeacherMove
from chessmaxx.training.memorize import (
    evaluate_memorization,
    summarize_memorization,
)
from chessmaxx.training.schema import TrainingExample


class FakeGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.offset = 0
        self.resets = 0

    @property
    def metadata(self) -> dict[str, str]:
        return {"model": "fake"}

    @property
    def telemetry(self) -> dict[str, int]:
        return {"positions_generated": self.offset}

    def reset_telemetry(self) -> None:
        self.offset = 0
        self.resets += 1

    def generate_many(self, fens: list[str]) -> list[GeneratedMove]:
        values = self.outputs[self.offset : self.offset + len(fens)]
        self.offset += len(fens)
        return [GeneratedMove(value, 1.0) for value in values]


def example(number: int, target: str) -> TrainingExample:
    board = chess.Board()
    return TrainingExample(
        example_id=f"example-{number}",
        game_id=f"game-{number}",
        ply=0,
        fen=board.fen(),
        target_move=target,
        teacher_moves=(TeacherMove(target, 20),),
        split="train",
        source="fixture.pgn",
    )


def test_memorization_distinguishes_move_and_response_matches() -> None:
    examples = [example(1, "e2e4"), example(2, "d2d4"), example(3, "g1f3")]
    generator = FakeGenerator(["e2e4", "d2d4 explanation", "h9h8"])

    results, telemetry = evaluate_memorization(
        examples, generator, batch_size=2
    )
    summary = summarize_memorization(results)

    assert generator.resets == 1
    assert telemetry == {"positions_generated": 3}
    assert summary == {
        "examples": 3,
        "parsed_moves": 2,
        "parse_rate": 2 / 3,
        "legal_moves": 2,
        "legal_move_rate": 2 / 3,
        "target_move_matches": 2,
        "target_move_accuracy": 2 / 3,
        "exact_response_matches": 1,
        "exact_response_accuracy": 1 / 3,
    }


def test_memorization_requires_a_positive_batch_size() -> None:
    try:
        evaluate_memorization([example(1, "e2e4")], FakeGenerator([]), batch_size=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected invalid batch size to fail")


def test_summarize_memorization_rejects_empty_results() -> None:
    try:
        summarize_memorization([])
    except ValueError as exc:
        assert "zero" in str(exc)
    else:
        raise AssertionError("expected empty results to fail")
