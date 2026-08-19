from pathlib import Path

import chess

from chessmaxx.evaluation.schema import TeacherMove
from chessmaxx.training.schema import TrainingExample
from chessmaxx.training.train import prepare_training_dataset, select_training_examples


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        tokens = [len(part) for part in text.split()]
        return ([1] if add_special_tokens else []) + tokens


def example(number: int, split: str = "train") -> TrainingExample:
    board = chess.Board()
    moves = list(board.legal_moves)
    return TrainingExample(
        example_id=f"example-{number}",
        game_id=f"game-{number}",
        ply=0,
        fen=board.fen(),
        target_move=moves[0].uci(),
        teacher_moves=(TeacherMove(moves[0].uci(), 20),),
        split=split,
        source="fixture.pgn",
    )


def test_select_training_examples_filters_and_caps() -> None:
    examples = [example(1), example(2, "validation"), example(3), example(4)]

    selected = select_training_examples(examples, maximum=2)

    assert [item.example_id for item in selected] == ["example-1", "example-3"]


def test_select_training_examples_requires_a_train_split() -> None:
    try:
        select_training_examples([example(1, "test")], maximum=100)
    except ValueError as exc:
        assert "no training examples" in str(exc)
    else:
        raise AssertionError("expected missing training data to fail")


def test_prepare_training_dataset_reports_packing_savings() -> None:
    examples = [example(1), example(2)]

    unpacked, unpacked_summary = prepare_training_dataset(
        examples, FakeTokenizer(), max_length=256, packing=False
    )
    packed, packed_summary = prepare_training_dataset(
        examples, FakeTokenizer(), max_length=256, packing=True
    )

    assert len(unpacked) == 2
    assert len(packed) == 1
    assert packed_summary.selected_examples == 2
    assert packed_summary.optimizer_records == 1
    assert packed_summary.input_tokens_per_epoch == unpacked_summary.input_tokens_per_epoch
    assert packed_summary.supervised_tokens_per_epoch == 4


def test_training_cli_help_does_not_import_gpu_dependencies() -> None:
    from chessmaxx.training.train_cli import main

    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
