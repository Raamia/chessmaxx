import chess
import pytest

from chessmaxx.evaluation.model import build_prompt
from chessmaxx.evaluation.schema import TeacherMove
from chessmaxx.training.schema import TrainingExample
from chessmaxx.training.tokenize import (
    IGNORE_INDEX,
    SupervisedTokenDataset,
    encode_training_example,
    format_target,
)


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, *, add_special_tokens):
        prefix = [1] if add_special_tokens else []
        return prefix + [ord(character) for character in text]


def example():
    return TrainingExample(
        example_id="start",
        game_id="game-1",
        ply=0,
        fen=chess.STARTING_FEN,
        target_move="e2e4",
        teacher_moves=(TeacherMove("e2e4", 20),),
        split="train",
        source="fixture.pgn",
    )


def test_only_target_and_eos_tokens_contribute_to_loss():
    tokenizer = CharacterTokenizer()
    encoded = encode_training_example(example(), tokenizer, max_length=256)
    prompt_length = len(tokenizer.encode(build_prompt(chess.STARTING_FEN), add_special_tokens=True))
    expected_target = tokenizer.encode(format_target("e2e4"), add_special_tokens=False) + [0]

    assert all(label == IGNORE_INDEX for label in encoded.labels[:prompt_length])
    assert list(encoded.labels[prompt_length:]) == expected_target
    assert encoded.supervised_tokens == len(expected_target)
    assert len(encoded.input_ids) == len(encoded.labels) == len(encoded.attention_mask)


def test_encoding_fails_instead_of_truncating_position_or_target():
    with pytest.raises(ValueError, match="exceeding max_length"):
        encode_training_example(example(), CharacterTokenizer(), max_length=8)


def test_map_dataset_returns_trainer_shaped_records():
    dataset = SupervisedTokenDataset(
        [example()], CharacterTokenizer(), max_length=256
    )

    assert len(dataset) == 1
    assert set(dataset[0]) == {"input_ids", "attention_mask", "labels"}
