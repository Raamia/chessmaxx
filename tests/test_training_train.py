from pathlib import Path

import chess

from chessmaxx.evaluation.schema import TeacherMove
from chessmaxx.training.config import TinySFTProfile
from chessmaxx.training.distillation import PolicyDatasetSummary
from chessmaxx.training.packing import IsolatedPackedTokenDataset
from chessmaxx.training.schema import TrainingExample
from chessmaxx.training.train import (
    _warmup_arguments,
    extract_learning_curve,
    policy_memory_estimates,
    prepare_policy_training_dataset,
    prepare_training_dataset,
    select_split_examples,
    select_training_examples,
)


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
        assert "contains 0 train examples" in str(exc)
    else:
        raise AssertionError("expected missing training data to fail")


def test_split_selection_requires_the_profile_count() -> None:
    try:
        select_split_examples(
            [example(1, "validation")], split="validation", maximum=2
        )
    except ValueError as exc:
        assert "contains 1 validation examples" in str(exc)
        assert "requires 2" in str(exc)
    else:
        raise AssertionError("expected an undersized split to fail")


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


def test_prepare_training_dataset_selects_isolated_packing():
    dataset, summary = prepare_training_dataset(
        [example(1), example(2)],
        FakeTokenizer(),
        max_length=256,
        packing=True,
        isolate_packed_attention=True,
    )

    assert isinstance(dataset, IsolatedPackedTokenDataset)
    assert summary.optimizer_records == 1


def test_prepare_policy_dataset_counts_all_candidate_sequences():
    dataset, summary = prepare_policy_training_dataset(
        [example(1), example(2)],
        FakeTokenizer(),
        max_length=256,
        temperature_cp=100.0,
        max_candidates=1,
    )

    assert len(dataset) == 2
    assert summary.optimizer_records == 2
    assert summary.supervised_tokens_per_epoch == 4


def test_warmup_arguments_supports_transformers_4_signature() -> None:
    def training_arguments(*, warmup_ratio: float = 0.0) -> None:
        pass

    assert _warmup_arguments(training_arguments, 0.05) == {"warmup_ratio": 0.05}


def test_warmup_arguments_supports_transformers_5_signature() -> None:
    def training_arguments(*, warmup_steps: int | float = 0) -> None:
        pass

    assert _warmup_arguments(training_arguments, 0.05) == {"warmup_steps": 0.05}


def test_warmup_arguments_rejects_unknown_signature() -> None:
    def training_arguments(*, learning_rate: float = 0.001) -> None:
        pass

    try:
        _warmup_arguments(training_arguments, 0.05)
    except RuntimeError as exc:
        assert "neither warmup_ratio nor warmup_steps" in str(exc)
    else:
        raise AssertionError("expected unsupported warmup arguments to fail")


def test_policy_memory_estimates_compare_dense_batch_with_one_chunk():
    profile = TinySFTProfile(
        name="chunked",
        model_id="model",
        objective="multipv_policy",
        packing=False,
        per_device_batch_size=2,
        max_teacher_candidates=3,
        policy_loss_backend="chunked_exact",
        vocabulary_chunk_size=40,
    )
    summary = PolicyDatasetSummary(
        examples=10,
        candidate_sequences=30,
        mean_candidates_per_example=3.0,
        mean_teacher_top1_probability=0.5,
        mean_teacher_entropy=0.9,
        maximum_candidate_length=96,
        maximum_supervised_tokens_per_example=15,
    )

    estimates = policy_memory_estimates(
        profile,
        summary,
        vocabulary_size=100,
        hidden_size=8,
        model_dtype_bytes=2,
    )

    assert estimates["dense_logits_bytes_per_longest_batch"] == 115_200
    assert estimates["chunked_supervised_tokens_per_longest_batch"] == 30
    assert estimates["chunked_float32_logits_bytes_per_chunk"] == 4_800
    assert estimates["chunked_saved_hidden_bytes_per_longest_batch"] == 480
    assert estimates["float32_logits_reduction_ratio"] == 48.0


def test_learning_curve_merges_train_and_validation_logs_by_step() -> None:
    curve = extract_learning_curve(
        [
            {"loss": 1.2, "learning_rate": 0.001, "epoch": 1.0, "step": 25},
            {"eval_loss": 1.1, "eval_runtime": 2.0, "epoch": 1.0, "step": 25},
            {"loss": 0.8, "learning_rate": 0.0005, "epoch": 2.0, "step": 50},
            {"train_runtime": 10.0, "step": 50},
        ]
    )

    assert curve == [
        {
            "step": 25,
            "loss": 1.2,
            "eval_loss": 1.1,
            "learning_rate": 0.001,
            "eval_runtime": 2.0,
            "epoch": 1.0,
        },
        {
            "step": 50,
            "loss": 0.8,
            "learning_rate": 0.0005,
            "epoch": 2.0,
        },
    ]


def test_training_cli_help_does_not_import_gpu_dependencies() -> None:
    from chessmaxx.training.train_cli import main

    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
