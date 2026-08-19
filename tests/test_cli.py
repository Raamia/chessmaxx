import json

import chess

from chessmaxx.evaluation import cli
from chessmaxx.evaluation.dataset import write_positions
from chessmaxx.evaluation.model import GeneratedMove
from chessmaxx.evaluation.schema import EvaluationPosition, TeacherMove
from chessmaxx.training.dataset import write_training_examples
from chessmaxx.training.schema import TrainingExample


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


def test_adapter_cli_evaluates_labelled_validation_split(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "training.jsonl"
    adapter = tmp_path / "adapter"
    output = tmp_path / "report.json"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    write_training_examples(
        dataset,
        [
            TrainingExample(
                example_id="validation-1",
                game_id="game-1",
                ply=0,
                fen=chess.STARTING_FEN,
                target_move="e2e4",
                teacher_moves=(TeacherMove("e2e4", 20),),
                split="validation",
                source="fixture.pgn",
            )
        ],
    )
    loaded = {}

    def load_adapter(path, **kwargs):
        loaded.update({"path": path, **kwargs})
        return FakeGenerator()

    monkeypatch.setattr(
        cli.HuggingFaceMoveGenerator, "from_adapter", load_adapter
    )
    monkeypatch.setattr(
        cli.StockfishAnalyzer,
        "open",
        lambda *args, **kwargs: FakeAnalyzer(),
    )

    assert (
        cli.main(
            [
                "adapter",
                "--training-profile",
                "configs/train/tiny-sft-qwen3-0.6b-unpacked.toml",
                "--adapter-dir",
                str(adapter),
                "--dataset",
                str(dataset),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["base_model_name"] == "Qwen/Qwen3-0.6B-Base"
    assert report["settings"]["mode"] == "adapter"
    assert report["settings"]["split"] == "validation"
    assert report["settings"]["adapter_sha256"]


def test_labelled_base_cli_uses_the_same_validation_split(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "training.jsonl"
    output = tmp_path / "report.json"
    write_training_examples(
        dataset,
        [
            TrainingExample(
                example_id="validation-1",
                game_id="game-1",
                ply=0,
                fen=chess.STARTING_FEN,
                target_move="e2e4",
                teacher_moves=(TeacherMove("e2e4", 20),),
                split="validation",
                source="fixture.pgn",
            )
        ],
    )
    loaded = {}

    def load_base(model_name, **kwargs):
        loaded.update({"model_name": model_name, **kwargs})
        return FakeGenerator()

    monkeypatch.setattr(
        cli.HuggingFaceMoveGenerator, "from_pretrained", load_base
    )
    monkeypatch.setattr(
        cli.StockfishAnalyzer,
        "open",
        lambda *args, **kwargs: FakeAnalyzer(),
    )

    assert (
        cli.main(
            [
                "labelled-base",
                "--training-profile",
                "configs/train/scaled-sft-qwen3-0.6b-isolated.toml",
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
    assert report["settings"]["mode"] == "labelled_base"
    assert report["settings"]["split"] == "validation"


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


def test_discovers_adapter_checkpoints_in_step_order(tmp_path):
    run_dir = tmp_path / "run"
    for relative in (
        "checkpoints/checkpoint-100",
        "checkpoints/checkpoint-25",
        "checkpoints/not-a-checkpoint",
        "final",
    ):
        directory = run_dir / relative
        directory.mkdir(parents=True)
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")

    discovered = cli.discover_adapter_checkpoints(run_dir)

    assert [(label, step) for label, _, step in discovered] == [
        ("checkpoint-25", 25),
        ("checkpoint-100", 100),
        ("final", None),
    ]
