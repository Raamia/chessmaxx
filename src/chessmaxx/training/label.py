"""Resumable conversion of chess positions into Stockfish supervision."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from chessmaxx.evaluation.schema import EvaluationPosition, TeacherMove
from chessmaxx.training.schema import (
    PROMPT_VERSION,
    TRAINING_SCHEMA_VERSION,
    TrainingExample,
)


class TeacherAnalyzer(Protocol):
    engine_id: dict[str, str]

    def analyze_fen(self, fen: str) -> tuple[TeacherMove, ...]: ...


class LabelJournalError(ValueError):
    pass


class LabelJournal:
    def __init__(self, path: str | Path, run_key: str) -> None:
        self.path = Path(path)
        self.run_key = run_key

    def load_or_create(self) -> dict[str, TrainingExample]:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            manifest = {
                "type": "manifest",
                "run_key": self.run_key,
                "created_at": datetime.now(UTC).isoformat(),
            }
            with self.path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return {}

        examples: dict[str, TrainingExample] = {}
        with self.path.open(encoding="utf-8") as handle:
            manifest = json.loads(handle.readline())
            if manifest.get("type") != "manifest":
                raise LabelJournalError(f"{self.path}: missing label manifest")
            if manifest.get("run_key") != self.run_key:
                raise LabelJournalError(
                    f"{self.path}: labels belong to a different run"
                )
            for line_number, line in enumerate(handle, start=2):
                try:
                    value = json.loads(line)
                    if value.get("type") != "example":
                        raise ValueError("unknown journal record")
                    example = TrainingExample.from_dict(value["value"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise LabelJournalError(
                        f"{self.path}:{line_number}: {exc}"
                    ) from exc
                if example.example_id in examples:
                    raise LabelJournalError(
                        f"{self.path}:{line_number}: duplicate example"
                    )
                examples[example.example_id] = example
        return examples

    def append(self, example: TrainingExample) -> None:
        record = {"type": "example", "value": example.to_dict()}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class TrainingLabeler:
    def __init__(
        self,
        analyzer: TeacherAnalyzer,
        split_assignments: dict[str, str],
    ) -> None:
        self.analyzer = analyzer
        self.split_assignments = split_assignments

    def _run_key(self, positions: Sequence[EvaluationPosition]) -> str:
        payload = {
            "positions": [
                {
                    "position_id": position.position_id,
                    "game_id": position.game_id,
                    "fen": position.fen,
                }
                for position in positions
            ],
            "splits": self.split_assignments,
            "engine": self.analyzer.engine_id,
            "schema_version": TRAINING_SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def run(
        self,
        positions: Sequence[EvaluationPosition],
        *,
        journal_path: str | Path | None = None,
    ) -> list[TrainingExample]:
        for position in positions:
            if position.game_id is None:
                raise ValueError(f"position {position.position_id!r} has no game_id")
            if position.game_id not in self.split_assignments:
                raise ValueError(f"game {position.game_id!r} has no split assignment")

        journal = None
        restored: dict[str, TrainingExample] = {}
        if journal_path is not None:
            journal = LabelJournal(journal_path, self._run_key(positions))
            restored = journal.load_or_create()

        examples = dict(restored)
        for position in positions:
            if position.position_id in examples:
                continue
            teacher_moves = position.teacher_moves or self.analyzer.analyze_fen(
                position.fen
            )
            if not teacher_moves:
                raise RuntimeError(
                    f"teacher returned no moves for {position.position_id!r}"
                )
            example = TrainingExample(
                example_id=position.position_id,
                game_id=position.game_id or "",
                ply=position.ply or 0,
                fen=position.fen,
                target_move=teacher_moves[0].move,
                teacher_moves=teacher_moves,
                split=self.split_assignments[position.game_id or ""],
                source=str(position.metadata.get("source", "unknown")),
            )
            examples[example.example_id] = example
            if journal is not None:
                journal.append(example)
        return [examples[position.position_id] for position in positions]

