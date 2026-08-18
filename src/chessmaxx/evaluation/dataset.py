"""JSONL persistence for frozen evaluation positions."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from chessmaxx.evaluation.schema import EvaluationPosition


class DatasetError(ValueError):
    """Raised when an evaluation dataset cannot be decoded or validated."""


def load_positions(path: str | Path) -> list[EvaluationPosition]:
    source = Path(path)
    positions: list[EvaluationPosition] = []
    seen_ids: set[str] = set()

    with source.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
                position = EvaluationPosition.from_dict(value)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DatasetError(f"{source}:{line_number}: {exc}") from exc
            if position.position_id in seen_ids:
                raise DatasetError(
                    f"{source}:{line_number}: duplicate position_id "
                    f"{position.position_id!r}"
                )
            seen_ids.add(position.position_id)
            positions.append(position)

    if not positions:
        raise DatasetError(f"{source}: dataset contains no positions")
    return positions


def write_positions(
    path: str | Path, positions: Iterable[EvaluationPosition]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for position in positions:
            handle.write(json.dumps(position.to_dict(), sort_keys=True) + "\n")

