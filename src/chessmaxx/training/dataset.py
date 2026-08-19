"""JSONL persistence for Stockfish-supervised examples."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from chessmaxx.training.schema import TrainingExample


class TrainingDatasetError(ValueError):
    pass


def load_training_examples(path: str | Path) -> list[TrainingExample]:
    source = Path(path)
    examples: list[TrainingExample] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                example = TrainingExample.from_dict(value)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TrainingDatasetError(f"{source}:{line_number}: {exc}") from exc
            if example.example_id in seen:
                raise TrainingDatasetError(
                    f"{source}:{line_number}: duplicate example_id {example.example_id!r}"
                )
            seen.add(example.example_id)
            examples.append(example)
    if not examples:
        raise TrainingDatasetError(f"{source}: dataset contains no examples")
    return examples


def write_training_examples(
    path: str | Path, examples: Iterable[TrainingExample]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + "\n")

