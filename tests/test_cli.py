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


def test_baseline_cli_uses_checked_in_profile(tmp_path, monkeypatch):
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
    loaded = {}

    def load_model(model_name, **kwargs):
        loaded.update({"model_name": model_name, **kwargs})
        return FakeGenerator()

    monkeypatch.setattr(
        cli.HuggingFaceMoveGenerator, "from_pretrained", load_model
    )
    monkeypatch.setattr(
        cli.StockfishAnalyzer,
        "open",
        lambda *args, **kwargs: FakeAnalyzer(),
    )

    assert (
        cli.main(
            [
                "baseline",
                "--profile",
                "configs/baseline/qwen3-0.6b-base.toml",
                "--dataset",
                str(dataset),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["model_name"] == "Qwen/Qwen3-0.6B-Base"
    assert loaded["revision"] == "main"
    assert report["settings"]["mode"] == "baseline"
    assert report["settings"]["profile_sha256"]


def test_sample_pgn_cli_writes_frozen_dataset(tmp_path):
    pgn = tmp_path / "game.pgn"
    output = tmp_path / "sample.jsonl"
    pgn.write_text(
        '[Event "Fixture"]\n[Result "*"]\n\n'
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *\n",
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "sample-pgn",
                "--pgn",
                str(pgn),
                "--output",
                str(output),
                "--count",
                "2",
                "--minimum-ply",
                "0",
            ]
        )
        == 0
    )
    assert len(cli.load_positions(output)) == 2
