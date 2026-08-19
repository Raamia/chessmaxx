import pytest

from chessmaxx.training.packing import pack_encoded_examples, pad_records
from chessmaxx.training.tokenize import EncodedExample, IGNORE_INDEX


def encoded(example_id, tokens, labels):
    return EncodedExample(
        example_id=example_id,
        input_ids=tuple(tokens),
        attention_mask=(1,) * len(tokens),
        labels=tuple(labels),
    )


def test_packing_preserves_order_targets_and_masked_boundaries():
    examples = [
        encoded("a", [1, 2, 3], [IGNORE_INDEX, 2, 3]),
        encoded("b", [4, 5], [IGNORE_INDEX, 5]),
        encoded("c", [6, 7, 8], [IGNORE_INDEX, 7, 8]),
    ]

    packed = pack_encoded_examples(examples, max_length=5)

    assert [item.example_ids for item in packed] == [("a", "b"), ("c",)]
    assert packed[0].input_ids == (1, 2, 3, 4, 5)
    assert packed[0].labels == (IGNORE_INDEX, 2, 3, IGNORE_INDEX, 5)
    assert sum(label != IGNORE_INDEX for item in packed for label in item.labels) == 5


def test_packing_rejects_oversized_examples():
    with pytest.raises(ValueError, match="exceeds packing"):
        pack_encoded_examples(
            [encoded("large", [1, 2, 3], [1, 2, 3])], max_length=2
        )


def test_padding_masks_pad_tokens_out_of_attention_and_loss():
    batch = pad_records(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ],
        pad_token_id=0,
    )

    assert batch["input_ids"] == [[1, 2], [3, 0]]
    assert batch["attention_mask"] == [[1, 1], [1, 0]]
    assert batch["labels"] == [[-100, 2], [3, -100]]
