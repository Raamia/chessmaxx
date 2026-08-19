import json

import chess

from chessmaxx.evaluation.model import GeneratedMove
from chessmaxx.evaluation.runner import EvaluationRunner
from chessmaxx.evaluation.schema import EvaluationPosition, TeacherMove


class FakeGenerator:
    metadata = {"model": "fixture-model", "decoding": "greedy"}

    def __init__(self):
        self.reset_count = 0

    def reset_telemetry(self):
        self.reset_count += 1

    @property
    def telemetry(self):
        return {"positions_per_second": 12.5}

    def generate_many(self, fens):
        outputs = ["e2e4", "e2e5", "g1f3"]
        return [GeneratedMove(output, 2.0) for output in outputs[: len(fens)]]


class FakeAnalyzer:
    engine_id = {"name": "fixture-engine"}

    def __init__(self):
        self.scored = []

    def analyze_fen(self, fen):
        return (
            TeacherMove("e2e4", 30),
            TeacherMove("d2d4", 20),
        )

    def score_move(self, fen, move_uci):
        self.scored.append(move_uci)
        return -90


def test_runner_preserves_raw_results_and_aggregates_metrics(tmp_path):
    positions = [
        EvaluationPosition(f"p-{index}", chess.STARTING_FEN)
        for index in range(3)
    ]
    analyzer = FakeAnalyzer()
    generator = FakeGenerator()
    runner = EvaluationRunner(generator, analyzer, batch_size=3)

    report = runner.run(positions)
    path = tmp_path / "report.json"
    report.write(path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert saved["summary"]["positions"] == 3
    assert saved["summary"]["parse_rate"] == 1.0
    assert saved["summary"]["legal_move_rate"] == 2 / 3
    assert saved["summary"]["top1_agreement_rate"] == 1 / 3
    assert saved["summary"]["average_centipawn_regret"] == 60
    assert analyzer.scored == ["g1f3"]
    assert saved["results"][1]["error"] == "illegal_move"
    assert saved["telemetry"]["positions_per_second"] == 12.5
    assert generator.reset_count == 1


def test_runner_restores_completed_positions_without_generating_again(tmp_path):
    positions = [
        EvaluationPosition(f"p-{index}", chess.STARTING_FEN)
        for index in range(3)
    ]
    journal = tmp_path / "progress.jsonl"
    first_generator = FakeGenerator()
    first = EvaluationRunner(
        first_generator, FakeAnalyzer(), batch_size=3, journal_path=journal
    ).run(positions)

    class ResumeGenerator(FakeGenerator):
        def generate_many(self, fens):
            raise AssertionError("completed positions should not be generated again")

    resumed = EvaluationRunner(
        ResumeGenerator(), FakeAnalyzer(), batch_size=3, journal_path=journal
    ).run(positions)

    assert resumed.results == first.results
    assert resumed.telemetry["positions_restored"] == 3
