import pytest

from chessmaxx.evaluation.metrics import PositionResult, summarize


def result(
    position_id,
    *,
    parsed_move=None,
    legal=False,
    teacher_moves=("e2e4",),
    regret=None,
    latency=1.0,
):
    return PositionResult(
        position_id=position_id,
        fen="fixture",
        raw_output=parsed_move or "bad",
        candidate=parsed_move,
        parsed_move=parsed_move,
        is_legal=legal,
        error=None if legal else "invalid_uci",
        teacher_moves=teacher_moves,
        best_score_cp=20,
        model_score_cp=None if regret is None else 20 - regret,
        centipawn_regret=regret,
        latency_ms=latency,
    )


def test_summary_uses_all_positions_for_agreement_and_scored_moves_for_blunders():
    results = [
        result("best", parsed_move="e2e4", legal=True, regret=0, latency=1),
        result("blunder", parsed_move="g1f3", legal=True, regret=350, latency=2),
        result("invalid", latency=30),
    ]

    summary = summarize(results)

    assert summary["legal_move_rate"] == pytest.approx(2 / 3)
    assert summary["top1_agreement_rate"] == pytest.approx(1 / 3)
    assert summary["blunder_300_rate"] == 0.5
    assert summary["average_centipawn_regret"] == 175
    assert summary["p95_latency_ms"] == 30


def test_summary_marks_engine_metrics_undefined_without_scored_moves():
    summary = summarize([result("invalid")])

    assert summary["scored_legal_moves"] == 0
    assert summary["average_centipawn_regret"] is None
    assert summary["blunder_100_rate"] is None
    assert summary["blunder_300_rate"] is None
    assert summary["blunder_500_rate"] is None
