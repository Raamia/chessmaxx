import json

import chess

from chessmaxx.evaluation.model import GeneratedMove
from chessmaxx.tournament import cli


class FirstLegalGenerator:
    metadata = {"kind": "fake-model", "revision": "fixture"}

    def reset_telemetry(self):
        self.positions = 0

    @property
    def telemetry(self):
        return {"positions_generated": self.positions}

    def generate_many(self, fens):
        self.positions += len(fens)
        return [
            GeneratedMove(
                sorted(move.uci() for move in chess.Board(fen).legal_moves)[0],
                latency_ms=1.0,
            )
            for fen in fens
        ]


def test_elo_cli_runs_and_resumes_smoke_tournament(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    report = tmp_path / "report.json"
    journal = tmp_path / "games.jsonl"
    pgn = tmp_path / "games.pgn"
    monkeypatch.setattr(
        cli.HuggingFaceLegalMoveRanker,
        "from_adapter",
        lambda *args, **kwargs: FirstLegalGenerator(),
    )
    arguments = [
        "--profile",
        "configs/elo/qwen3-0.6b-elo-smoke.toml",
        "--adapter-dir",
        str(adapter),
        "--openings",
        "data/elo/openings-v1.jsonl",
        "--report",
        str(report),
        "--journal",
        str(journal),
        "--pgn",
        str(pgn),
    ]

    assert cli.main(arguments) == 0
    first = json.loads(report.read_text(encoding="utf-8"))
    assert len(first["games"]) == 4
    assert first["games_restored"] == 0
    assert pgn.read_text(encoding="utf-8").count(
        '[Event "Chessmaxx Elo Evaluation"]'
    ) == 4

    assert cli.main(arguments) == 0
    resumed = json.loads(report.read_text(encoding="utf-8"))
    assert resumed["games_restored"] == 4
    assert journal.read_text(encoding="utf-8").count("\n") == 5
