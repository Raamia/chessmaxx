import chess
import pytest

from chessmaxx.evaluation.schema import TeacherMove
from chessmaxx.training.dataset import (
    TrainingDatasetError,
    evaluation_positions_for_split,
    load_training_examples,
    write_training_examples,
)
from chessmaxx.training.schema import TrainingExample


def example(**overrides):
    values = {
        "example_id": "game-1-ply-0",
        "game_id": "game-1",
        "ply": 0,
        "fen": chess.STARTING_FEN,
        "target_move": "e2e4",
        "teacher_moves": (
            TeacherMove("e2e4", 30),
            TeacherMove("d2d4", 20),
        ),
        "split": "train",
        "source": "fixture.pgn",
    }
    values.update(overrides)
    return TrainingExample(**values)


def test_training_examples_round_trip(tmp_path):
    path = tmp_path / "train.jsonl"
    expected = example()

    write_training_examples(path, [expected])

    assert load_training_examples(path) == [expected]


def test_target_must_match_top_teacher_move():
    with pytest.raises(ValueError, match="highest-ranked"):
        example(target_move="d2d4")


def test_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "train.jsonl"
    write_training_examples(path, [example(), example()])

    with pytest.raises(TrainingDatasetError, match="duplicate example_id"):
        load_training_examples(path)


def test_projects_one_training_split_into_evaluation_positions():
    train = example(example_id="train-1")
    validation = example(example_id="validation-1", split="validation")

    positions = evaluation_positions_for_split([train, validation], "validation")

    assert len(positions) == 1
    assert positions[0].position_id == "validation-1"
    assert positions[0].teacher_moves == validation.teacher_moves
    assert positions[0].metadata["training_split"] == "validation"
    assert positions[0].metadata["prompt_version"] == validation.prompt_version


def test_split_projection_rejects_missing_split():
    with pytest.raises(TrainingDatasetError, match="no 'test' examples"):
        evaluation_positions_for_split([example()], "test")
