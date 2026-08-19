"""JSONL persistence for Stockfish-supervised examples."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from chessmaxx.evaluation.schema import EvaluationPosition
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


def evaluation_positions_for_split(
    examples: Iterable[TrainingExample], split: str
) -> list[EvaluationPosition]:
    """Project one labelled split into the frozen-position evaluation schema."""

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    positions = [
        EvaluationPosition(
            position_id=example.example_id,
            fen=example.fen,
            game_id=example.game_id,
            ply=example.ply,
            teacher_moves=example.teacher_moves,
            metadata={
                "source": example.source,
                "training_split": example.split,
                "training_schema_version": example.schema_version,
                "prompt_version": example.prompt_version,
            },
        )
        for example in examples
        if example.split == split
    ]
    if not positions:
        raise TrainingDatasetError(f"dataset contains no {split!r} examples")
    return positions
