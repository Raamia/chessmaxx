import math

import chess
import pytest

from chessmaxx.evaluation.schema import TeacherMove
from chessmaxx.training.distillation import (
    PolicyTokenDataset,
    centipawn_policy,
    encode_policy_example,
    pad_policy_records,
    policy_entropy,
    summarize_policy_dataset,
)
from chessmaxx.training.schema import TrainingExample
from chessmaxx.training.tokenize import IGNORE_INDEX


MOVES = (
    TeacherMove("e2e4", 80),
    TeacherMove("d2d4", 40),
    TeacherMove("g1f3", 0),
)


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, *, add_special_tokens):
        tokens = [len(part) for part in text.split()]
        return ([1] if add_special_tokens else []) + tokens


def example():
    return TrainingExample(
        example_id="position-1",
        game_id="game-1",
        ply=0,
        fen=chess.STARTING_FEN,
        target_move="e2e4",
        teacher_moves=MOVES,
        split="train",
        source="fixture.pgn",
    )


def test_centipawn_policy_is_normalized_and_ranked():
    policy = centipawn_policy(MOVES, temperature_cp=100.0, max_candidates=3)

    assert [target.move for target in policy] == ["e2e4", "d2d4", "g1f3"]
    assert sum(target.probability for target in policy) == pytest.approx(1.0)
    assert policy[0].probability > policy[1].probability > policy[2].probability


def test_temperature_controls_teacher_sharpness():
    cold = centipawn_policy(MOVES, temperature_cp=25.0, max_candidates=3)
    warm = centipawn_policy(MOVES, temperature_cp=200.0, max_candidates=3)

    assert cold[0].probability > warm[0].probability
    assert policy_entropy(cold) < policy_entropy(warm)


def test_truncation_renormalizes_the_retained_candidates():
    policy = centipawn_policy(MOVES, temperature_cp=100.0, max_candidates=2)

    assert len(policy) == 2
    assert sum(target.probability for target in policy) == pytest.approx(1.0)


def test_equal_scores_produce_a_uniform_policy():
    policy = centipawn_policy(
        (TeacherMove("e2e4", 10), TeacherMove("d2d4", 10)),
        temperature_cp=50.0,
        max_candidates=2,
    )

    assert [target.probability for target in policy] == pytest.approx([0.5, 0.5])
    assert policy_entropy(policy) == pytest.approx(math.log(2))


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_policy_rejects_nonpositive_temperature(temperature):
    with pytest.raises(ValueError, match="temperature_cp"):
        centipawn_policy(MOVES, temperature_cp=temperature, max_candidates=3)


def test_encodes_each_policy_move_against_the_same_prompt():
    encoded = encode_policy_example(
        example(),
        FakeTokenizer(),
        max_length=256,
        temperature_cp=100.0,
        max_candidates=3,
    )

    assert len(encoded.candidates) == 3
    assert encoded.candidates[0].input_ids[:-2] == encoded.candidates[1].input_ids[:-2]
    assert encoded.candidates[0].labels[-2:] == (4, 99)
    assert all(
        label == IGNORE_INDEX for label in encoded.candidates[0].labels[:-2]
    )
    assert sum(encoded.teacher_probabilities) == pytest.approx(1.0)


def test_policy_dataset_preserves_candidate_groups():
    dataset = PolicyTokenDataset(
        [example()],
        FakeTokenizer(),
        max_length=256,
        temperature_cp=100.0,
        max_candidates=2,
    )

    item = dataset[0]
    assert item["example_id"] == "position-1"
    assert len(item["input_ids"]) == 2
    assert len(item["teacher_probabilities"]) == 2


def test_policy_collation_pads_candidates_and_tokens_independently():
    batch = pad_policy_records(
        [
            {
                "input_ids": [[1, 2, 3], [1, 4]],
                "attention_mask": [[1, 1, 1], [1, 1]],
                "labels": [[-100, 2, 3], [-100, 4]],
                "teacher_probabilities": [0.7, 0.3],
            },
            {
                "input_ids": [[5, 6]],
                "attention_mask": [[1, 1]],
                "labels": [[-100, 6]],
                "teacher_probabilities": [1.0],
            },
        ],
        pad_token_id=0,
    )

    assert batch["input_ids"] == [
        [[1, 2, 3], [1, 4, 0]],
        [[5, 6, 0], [0, 0, 0]],
    ]
    assert batch["labels"][1][1] == [-100, -100, -100]
    assert batch["teacher_probabilities"] == [[0.7, 0.3], [1.0, 0.0]]
    assert batch["candidate_mask"] == [[True, True], [True, False]]


def test_policy_dataset_summary_measures_teacher_and_candidate_shape():
    dataset = PolicyTokenDataset(
        [example(), example()],
        FakeTokenizer(),
        max_length=256,
        temperature_cp=100.0,
        max_candidates=3,
    )

    summary = summarize_policy_dataset(dataset)

    assert summary.examples == 2
    assert summary.candidate_sequences == 6
    assert summary.mean_candidates_per_example == 3.0
    assert 0 < summary.mean_teacher_top1_probability < 1
    assert summary.mean_teacher_entropy > 0
    assert summary.maximum_candidate_length > 0


def test_policy_dataset_summary_treats_underflowed_probability_as_zero_entropy():
    sharp_example = TrainingExample(
        example_id="sharp-position",
        game_id="game-1",
        ply=0,
        fen=chess.STARTING_FEN,
        target_move="e2e4",
        teacher_moves=(
            TeacherMove("e2e4", 100_000),
            TeacherMove("d2d4", -100_000),
            TeacherMove("g1f3", -100_000),
        ),
        split="train",
        source="fixture.pgn",
    )
    dataset = PolicyTokenDataset(
        [sharp_example],
        FakeTokenizer(),
        max_length=256,
        temperature_cp=100.0,
        max_candidates=3,
    )

    assert dataset.items[0].teacher_probabilities == (1.0, 0.0, 0.0)
    assert summarize_policy_dataset(dataset).mean_teacher_entropy == 0.0
