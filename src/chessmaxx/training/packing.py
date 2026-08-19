"""Greedy sequence packing and causal-LM batch collation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chessmaxx.training.tokenize import EncodedExample, IGNORE_INDEX


@dataclass(frozen=True, slots=True)
class PackedExample:
    example_ids: tuple[str, ...]
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]


def pack_encoded_examples(
    examples: list[EncodedExample], *, max_length: int
) -> list[PackedExample]:
    """Greedily concatenate complete examples without truncation or reordering."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    packed: list[PackedExample] = []
    current_ids: list[str] = []
    current_input: list[int] = []
    current_attention: list[int] = []
    current_labels: list[int] = []

    def flush() -> None:
        if not current_ids:
            return
        packed.append(
            PackedExample(
                example_ids=tuple(current_ids),
                input_ids=tuple(current_input),
                attention_mask=tuple(current_attention),
                labels=tuple(current_labels),
            )
        )
        current_ids.clear()
        current_input.clear()
        current_attention.clear()
        current_labels.clear()

    for example in examples:
        length = len(example.input_ids)
        if length > max_length:
            raise ValueError(
                f"example {example.example_id!r} exceeds packing max_length"
            )
        if current_input and len(current_input) + length > max_length:
            flush()
        current_ids.append(example.example_id)
        current_input.extend(example.input_ids)
        current_attention.extend(example.attention_mask)
        current_labels.extend(example.labels)
    flush()
    return packed


def pad_records(
    records: list[dict[str, list[int]]], *, pad_token_id: int
) -> dict[str, list[list[int]]]:
    if not records:
        raise ValueError("cannot collate an empty batch")
    width = max(len(record["input_ids"]) for record in records)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for record in records:
        padding = width - len(record["input_ids"])
        batch["input_ids"].append(record["input_ids"] + [pad_token_id] * padding)
        batch["attention_mask"].append(record["attention_mask"] + [0] * padding)
        batch["labels"].append(record["labels"] + [IGNORE_INDEX] * padding)
    return batch


class PackedTokenDataset:
    def __init__(self, examples: list[PackedExample]) -> None:
        self.items = examples

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        item = self.items[index]
        return {
            "input_ids": list(item.input_ids),
            "attention_mask": list(item.attention_mask),
            "labels": list(item.labels),
        }


class CausalLMCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, records: list[dict[str, list[int]]]) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("training requires the optional model dependencies") from exc
        padded = pad_records(records, pad_token_id=self.pad_token_id)
        return {name: torch.tensor(value, dtype=torch.long) for name, value in padded.items()}

