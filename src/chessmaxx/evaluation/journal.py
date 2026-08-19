"""Append-only progress journals for interruption-safe evaluations."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from chessmaxx.evaluation.metrics import PositionResult


class JournalError(ValueError):
    """Raised when progress belongs to a different or corrupted run."""


class ResultJournal:
    def __init__(self, path: str | Path, run_key: str) -> None:
        self.path = Path(path)
        self.run_key = run_key

    def load_or_create(self) -> dict[str, PositionResult]:
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

        results: dict[str, PositionResult] = {}
        saw_manifest = False
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise JournalError(
                        f"{self.path}:{line_number}: corrupted JSON"
                    ) from exc
                if line_number == 1:
                    if value.get("type") != "manifest":
                        raise JournalError(f"{self.path}: missing journal manifest")
                    if value.get("run_key") != self.run_key:
                        raise JournalError(
                            f"{self.path}: progress belongs to a different run"
                        )
                    saw_manifest = True
                    continue
                if value.get("type") != "result":
                    raise JournalError(
                        f"{self.path}:{line_number}: unknown journal record"
                    )
                result = PositionResult.from_dict(value["value"])
                if result.position_id in results:
                    raise JournalError(
                        f"{self.path}:{line_number}: duplicate position result"
                    )
                results[result.position_id] = result
        if not saw_manifest:
            raise JournalError(f"{self.path}: missing journal manifest")
        return results

    def append(self, result: PositionResult) -> None:
        record = {"type": "result", "value": result.to_dict()}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
