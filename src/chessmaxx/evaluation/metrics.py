"""Aggregation of position-level chess evaluation results."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PositionResult:
    position_id: str
    fen: str
    raw_output: str
    candidate: str | None
    parsed_move: str | None
    is_legal: bool
    error: str | None
    teacher_moves: tuple[str, ...]
    best_score_cp: int
    model_score_cp: int | None
    centipawn_regret: int | None
    latency_ms: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["teacher_moves"] = list(self.teacher_moves)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PositionResult":
        return cls(
            position_id=str(value["position_id"]),
            fen=str(value["fen"]),
            raw_output=str(value["raw_output"]),
            candidate=value.get("candidate"),
            parsed_move=value.get("parsed_move"),
            is_legal=bool(value["is_legal"]),
            error=value.get("error"),
            teacher_moves=tuple(value["teacher_moves"]),
            best_score_cp=int(value["best_score_cp"]),
            model_score_cp=value.get("model_score_cp"),
            centipawn_regret=value.get("centipawn_regret"),
            latency_ms=float(value["latency_ms"]),
            prompt_tokens=value.get("prompt_tokens"),
            output_tokens=value.get("output_tokens"),
        )


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize(results: Sequence[PositionResult]) -> dict[str, int | float]:
    total = len(results)
    parsed = sum(result.parsed_move is not None for result in results)
    legal = sum(result.is_legal for result in results)
    top1 = sum(
        result.is_legal
        and bool(result.teacher_moves)
        and result.parsed_move == result.teacher_moves[0]
        for result in results
    )
    topk = sum(
        result.is_legal and result.parsed_move in result.teacher_moves
        for result in results
    )
    regrets = [
        result.centipawn_regret
        for result in results
        if result.centipawn_regret is not None
    ]
    latencies = [result.latency_ms for result in results]

    return {
        "positions": total,
        "parsed_moves": parsed,
        "legal_moves": legal,
        "scored_legal_moves": len(regrets),
        "parse_rate": _rate(parsed, total),
        "legal_move_rate": _rate(legal, total),
        "top1_agreement_rate": _rate(top1, total),
        "topk_agreement_rate": _rate(topk, total),
        "average_centipawn_regret": statistics.fmean(regrets) if regrets else 0.0,
        "blunder_100_rate": _rate(sum(value >= 100 for value in regrets), len(regrets)),
        "blunder_300_rate": _rate(sum(value >= 300 for value in regrets), len(regrets)),
        "blunder_500_rate": _rate(sum(value >= 500 for value in regrets), len(regrets)),
        "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
    }
