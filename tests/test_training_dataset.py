import chess
import pytest

from chessmaxx.evaluation.schema import TeacherMove
from chessmaxx.training.dataset import (
    TrainingDatasetError,
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
