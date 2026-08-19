import chess
import pytest

from chessmaxx.evaluation.schema import TeacherMove
from chessmaxx.training.schema import TrainingExample
from chessmaxx.training.split import split_game_ids, validate_game_isolation


def example(example_id, split):
    return TrainingExample(
        example_id=example_id,
        game_id="shared-game",
        ply=0,
        fen=chess.STARTING_FEN,
        target_move="e2e4",
        teacher_moves=(TeacherMove("e2e4", 20),),
        split=split,
        source="fixture.pgn",
    )


def test_game_split_is_deterministic_and_input_order_independent():
    game_ids = [f"game-{index}" for index in range(20)]

    first = split_game_ids(
        game_ids, seed=42, validation_fraction=0.2, test_fraction=0.1
    )
    second = split_game_ids(
        reversed(game_ids), seed=42, validation_fraction=0.2, test_fraction=0.1
    )

    assert first == second
    assert list(first.values()).count("train") == 14
    assert list(first.values()).count("validation") == 4
    assert list(first.values()).count("test") == 2


def test_small_dataset_keeps_at_least_one_training_game():
    assignments = split_game_ids(
        ["game-a", "game-b"],
        validation_fraction=0.5,
        test_fraction=0.25,
    )

    assert "train" in assignments.values()


def test_game_isolation_rejects_cross_split_leakage():
    examples = [
        example("train", "train"),
        example("validation", "validation"),
    ]

    with pytest.raises(ValueError, match="appears in both"):
        validate_game_isolation(examples)
