"""Response-only tokenization for UCI move supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from chessmaxx.evaluation.model import build_prompt
from chessmaxx.training.schema import TrainingExample


IGNORE_INDEX = -100


class TokenizerLike(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class EncodedExample:
    example_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]

    @property
    def supervised_tokens(self) -> int:
        return sum(label != IGNORE_INDEX for label in self.labels)


def format_target(move_uci: str) -> str:
    """Include the generation-leading space learned after the `Move:` prompt."""

    return f" {move_uci}"


def encode_training_example(
    example: TrainingExample,
    tokenizer: TokenizerLike,
    *,
    max_length: int,
) -> EncodedExample:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    prompt_ids = tokenizer.encode(
        build_prompt(example.fen), add_special_tokens=True
    )
    target_ids = tokenizer.encode(
        format_target(example.target_move), add_special_tokens=False
    )
    if not target_ids:
        raise ValueError("target move encoded to zero tokens")
    response_ids = [*target_ids, tokenizer.eos_token_id]
    input_ids = [*prompt_ids, *response_ids]
    if len(input_ids) > max_length:
        raise ValueError(
            f"example {example.example_id!r} needs {len(input_ids)} tokens, "
            f"exceeding max_length={max_length}"
        )
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    return EncodedExample(
        example_id=example.example_id,
        input_ids=tuple(input_ids),
        attention_mask=(1,) * len(input_ids),
        labels=tuple(labels),
    )


class SupervisedTokenDataset:
    """Minimal map-style dataset accepted by Transformers Trainer."""

    def __init__(
        self,
        examples: list[TrainingExample],
        tokenizer: TokenizerLike,
        *,
        max_length: int,
    ) -> None:
        self.items = [
            encode_training_example(example, tokenizer, max_length=max_length)
            for example in examples
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {
            "input_ids": list(item.input_ids),
            "attention_mask": list(item.attention_mask),
            "labels": list(item.labels),
        }

