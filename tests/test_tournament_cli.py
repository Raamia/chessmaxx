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

    def generate_prompts(self, prompts):
        fens = [
            next(
                line.removeprefix("FEN: ")
                for line in prompt.splitlines()
                if line.startswith("FEN: ")
            )
            for prompt in prompts
        ]
        return self.generate_many(fens)


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
    assert first["invocation"]["games_generated"] == 4
    assert first["invocation"]["benchmark_telemetry_preserved"] is False
    first_telemetry = first["telemetry"]
    first_benchmark_created_at = first["benchmark_created_at"]
    assert pgn.read_text(encoding="utf-8").count(
        '[Event "Chessmaxx Elo Evaluation"]'
    ) == 4

    assert cli.main(arguments) == 0
    resumed = json.loads(report.read_text(encoding="utf-8"))
    assert resumed["games_restored"] == 4
    assert resumed["telemetry"] == first_telemetry
    assert resumed["benchmark_created_at"] == first_benchmark_created_at
    assert resumed["invocation"]["games_generated"] == 0
    assert resumed["invocation"]["benchmark_telemetry_preserved"] is True
    assert resumed["invocation"]["telemetry"]["chessmaxx"][
        "positions_generated"
    ] == 0
    assert journal.read_text(encoding="utf-8").count("\n") == 5


def test_elo_cli_can_evaluate_the_pinned_base_model(tmp_path, monkeypatch):
    report = tmp_path / "base-report.json"
    monkeypatch.setattr(
        cli.HuggingFaceLegalMoveRanker,
        "from_pretrained",
        lambda *args, **kwargs: FirstLegalGenerator(),
    )

    assert cli.main(
        [
            "--profile",
            "configs/elo/qwen3-0.6b-elo-smoke.toml",
            "--base-model-only",
            "--openings",
            "data/elo/openings-v1.jsonl",
            "--report",
            str(report),
            "--journal",
            str(tmp_path / "base-games.jsonl"),
            "--pgn",
            str(tmp_path / "base-games.pgn"),
        ]
    ) == 0

    value = json.loads(report.read_text(encoding="utf-8"))
    assert value["settings"]["model_source"] == "base"
    assert value["settings"]["adapter_sha256"] is None
    assert len(value["games"]) == 4


def test_elo_cli_wires_retry_and_history_modes(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    report = tmp_path / "assisted-report.json"
    monkeypatch.setattr(
        cli.HuggingFaceMoveGenerator,
        "from_adapter",
        lambda *args, **kwargs: FirstLegalGenerator(),
    )

    assert cli.main(
        [
            "--profile",
            "configs/elo/qwen3-0.6b-assisted-smoke.toml",
            "--adapter-dir",
            str(adapter),
            "--openings",
            "data/elo/openings-v1.jsonl",
            "--selection",
            "retry-with-legal-list",
            "--context",
            "fen-pgn",
            "--report",
            str(report),
            "--journal",
            str(tmp_path / "assisted-games.jsonl"),
            "--pgn",
            str(tmp_path / "assisted-games.pgn"),
        ]
    ) == 0

    settings = json.loads(report.read_text(encoding="utf-8"))["settings"]
    assert settings["selection"] == "retry-with-legal-list"
    assert settings["context"] == "fen-pgn"
    assert settings["max_attempts"] == 3
