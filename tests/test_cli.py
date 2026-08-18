import json

import chess

from chessmaxx.evaluation import cli
from chessmaxx.evaluation.dataset import write_positions
from chessmaxx.evaluation.model import GeneratedMove
from chessmaxx.evaluation.schema import EvaluationPosition, TeacherMove


class FakeGenerator:
    metadata = {"model": "fake-model"}

    def generate_many(self, fens):
        return [GeneratedMove("e2e4", 1.0) for _ in fens]


class FakeAnalyzer:
    engine_id = {"name": "Fakefish"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def analyze_fen(self, fen):
        raise AssertionError("fixture contains teacher moves")

    def score_move(self, fen, move_uci):
        raise AssertionError("generated move is already a teacher move")


def test_positions_cli_writes_reproducible_report(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "positions.jsonl"
    output = tmp_path / "report.json"
    write_positions(
        dataset,
        [
            EvaluationPosition(
                "start",
                chess.STARTING_FEN,
                teacher_moves=(TeacherMove("e2e4", 20),),
            )
        ],
    )
    monkeypatch.setattr(
        cli.HuggingFaceMoveGenerator,
        "from_pretrained",
        lambda *args, **kwargs: FakeGenerator(),
    )
    monkeypatch.setattr(
        cli.StockfishAnalyzer,
        "open",
        lambda *args, **kwargs: FakeAnalyzer(),
    )

    exit_code = cli.main(
        [
            "positions",
            "--model",
            "fake-model",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["summary"]["legal_move_rate"] == 1.0
    assert report["settings"]["dataset_sha256"]
    assert "legal_move_rate" in capsys.readouterr().out


def test_packaged_smoke_dataset_is_valid():
    positions = cli.load_positions("data/eval/smoke.jsonl")

    assert len(positions) == 3

