"""End-to-end orchestration for frozen-position evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from chessmaxx.evaluation.metrics import PositionResult, summarize
from chessmaxx.evaluation.journal import ResultJournal
from chessmaxx.evaluation.model import MoveGenerator
from chessmaxx.evaluation.moves import check_generated_move
from chessmaxx.evaluation.schema import EvaluationPosition, TeacherMove


class PositionAnalyzer(Protocol):
    engine_id: dict[str, str]

    def analyze_fen(self, fen: str) -> tuple[TeacherMove, ...]: ...

    def score_move(self, fen: str, move_uci: str) -> int: ...


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    created_at: str
    model: dict[str, Any]
    engine: dict[str, str]
    settings: dict[str, Any]
    telemetry: dict[str, Any]
    summary: dict[str, int | float]
    results: tuple[PositionResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "model": self.model,
            "engine": self.engine,
            "settings": self.settings,
            "telemetry": self.telemetry,
            "summary": self.summary,
            "results": [result.to_dict() for result in self.results],
        }

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class EvaluationRunner:
    def __init__(
        self,
        generator: MoveGenerator,
        analyzer: PositionAnalyzer,
        batch_size: int = 8,
        settings: dict[str, Any] | None = None,
        journal_path: str | Path | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.generator = generator
        self.analyzer = analyzer
        self.batch_size = batch_size
        self.settings = settings or {}
        self.journal_path = Path(journal_path) if journal_path is not None else None

    def _run_key(
        self,
        positions: Sequence[EvaluationPosition],
        model_metadata: dict[str, Any],
    ) -> str:
        payload = {
            "positions": [
                {"position_id": position.position_id, "fen": position.fen}
                for position in positions
            ],
            "model": model_metadata,
            "engine": self.analyzer.engine_id,
            "settings": {**self.settings, "batch_size": self.batch_size},
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def run(self, positions: Sequence[EvaluationPosition]) -> EvaluationReport:
        model_metadata = dict(self.generator.metadata)
        journal = None
        restored: dict[str, PositionResult] = {}
        if self.journal_path is not None:
            journal = ResultJournal(
                self.journal_path, self._run_key(positions, model_metadata)
            )
            restored = journal.load_or_create()
            expected = {position.position_id: position.fen for position in positions}
            for position_id, result in restored.items():
                if expected.get(position_id) != result.fen:
                    raise ValueError(
                        f"journal result {position_id!r} does not match the dataset"
                    )
        reset_telemetry = getattr(self.generator, "reset_telemetry", None)
        if reset_telemetry is not None:
            reset_telemetry()
        results_by_id = dict(restored)
        pending = [
            position for position in positions if position.position_id not in restored
        ]
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            generated = self.generator.generate_many(
                [position.fen for position in batch]
            )
            if len(generated) != len(batch):
                raise RuntimeError("model returned a different number of outputs than inputs")
            for position, response in zip(batch, generated, strict=True):
                checked = check_generated_move(position.fen, response.raw_output)
                teacher = position.teacher_moves or self.analyzer.analyze_fen(position.fen)
                best_score = teacher[0].score_cp
                model_score = None
                regret = None
                if checked.is_legal and checked.parsed_move is not None:
                    known = next(
                        (item for item in teacher if item.move == checked.parsed_move),
                        None,
                    )
                    model_score = (
                        known.score_cp
                        if known is not None
                        else self.analyzer.score_move(position.fen, checked.parsed_move)
                    )
                    regret = max(0, best_score - model_score)
                result = PositionResult(
                    position_id=position.position_id,
                    fen=position.fen,
                    raw_output=response.raw_output,
                    candidate=checked.candidate,
                    parsed_move=checked.parsed_move,
                    is_legal=checked.is_legal,
                    error=checked.error,
                    teacher_moves=tuple(item.move for item in teacher),
                    best_score_cp=best_score,
                    model_score_cp=model_score,
                    centipawn_regret=regret,
                    latency_ms=response.latency_ms,
                    prompt_tokens=response.prompt_tokens,
                    output_tokens=response.output_tokens,
                )
                results_by_id[position.position_id] = result
                if journal is not None:
                    journal.append(result)

        results = tuple(results_by_id[position.position_id] for position in positions)
        telemetry = dict(getattr(self.generator, "telemetry", {}))
        telemetry["positions_restored"] = len(restored)

        return EvaluationReport(
            created_at=datetime.now(UTC).isoformat(),
            model=model_metadata,
            engine=dict(self.analyzer.engine_id),
            settings={**self.settings, "batch_size": self.batch_size},
            telemetry=telemetry,
            summary=summarize(results),
            results=results,
        )
