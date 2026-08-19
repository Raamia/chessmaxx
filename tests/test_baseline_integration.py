import json

import chess

from chessmaxx.evaluation import cli
from chessmaxx.evaluation.model import GeneratedMove
from chessmaxx.evaluation.schema import TeacherMove


PGN = """[Event "Baseline Fixture"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 *
"""


class LegalMoveGenerator:
    metadata = {
        "model": "fixture-model",
        "requested_revision": "main",
        "resolved_revision": "fixture-revision",
    }

    def __init__(self, fail_if_called=False):
        self.fail_if_called = fail_if_called
        self.generated = 0

    def reset_telemetry(self):
        self.generated = 0

    @property
    def telemetry(self):
        return {"positions_generated": self.generated}

    def generate_many(self, fens):
        if self.fail_if_called:
            raise AssertionError("resume should not call the model")
        responses = []
        for fen in fens:
            move = next(iter(chess.Board(fen).legal_moves)).uci()
            responses.append(GeneratedMove(move, 1.0, 20, 1))
        self.generated += len(responses)
        return responses


class MatchingAnalyzer:
    engine_id = {"name": "Fixturefish 1"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def analyze_fen(self, fen):
        move = next(iter(chess.Board(fen).legal_moves)).uci()
        return (TeacherMove(move, 25),)

    def score_move(self, fen, move_uci):
        return 25


def test_sample_evaluate_report_and_resume_work_as_one_pipeline(tmp_path, monkeypatch):
    pgn = tmp_path / "games.pgn"
    dataset = tmp_path / "positions.jsonl"
    profile = tmp_path / "profile.toml"
    report_path = tmp_path / "baseline.json"
    pgn.write_text(PGN, encoding="utf-8")
    profile.write_text(
        """[baseline]
name = "fixture"
model_id = "fixture/model"
batch_size = 2
max_new_tokens = 8
stockfish_nodes = 100
stockfish_multipv = 1
stockfish_threads = 1
stockfish_hash_mb = 16
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli.StockfishAnalyzer,
        "open",
        lambda *args, **kwargs: MatchingAnalyzer(),
    )

    assert (
        cli.main(
            [
                "sample-pgn",
                "--pgn",
                str(pgn),
                "--output",
                str(dataset),
                "--count",
                "3",
                "--minimum-ply",
                "0",
            ]
        )
        == 0
    )
    monkeypatch.setattr(
        cli.HuggingFaceMoveGenerator,
        "from_pretrained",
        lambda *args, **kwargs: LegalMoveGenerator(),
    )
    baseline_args = [
        "baseline",
        "--profile",
        str(profile),
        "--dataset",
        str(dataset),
        "--output",
        str(report_path),
    ]

    assert cli.main(baseline_args) == 0
    first_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert first_report["summary"]["positions"] == 3
    assert first_report["summary"]["legal_move_rate"] == 1.0
    assert first_report["summary"]["top1_agreement_rate"] == 1.0
    assert first_report["telemetry"]["positions_generated"] == 3

    monkeypatch.setattr(
        cli.HuggingFaceMoveGenerator,
        "from_pretrained",
        lambda *args, **kwargs: LegalMoveGenerator(fail_if_called=True),
    )
    assert cli.main(baseline_args) == 0
    resumed_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert resumed_report["summary"] == first_report["summary"]
    assert resumed_report["telemetry"]["positions_restored"] == 3
