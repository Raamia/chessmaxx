import json

import chess

from chessmaxx.evaluation.schema import TeacherMove
from chessmaxx.training import cli
from chessmaxx.training.dataset import load_training_examples
from chessmaxx.training.split import validate_game_isolation


PGN = """[Event "One"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *

[Event "Two"]
[Result "*"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 *
"""


class FakeAnalyzer:
    engine_id = {"name": "Fixturefish"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def analyze_fen(self, fen):
        move = next(iter(chess.Board(fen).legal_moves)).uci()
        return (TeacherMove(move, 20),)


def test_build_command_writes_labeled_isolated_dataset_and_manifest(
    tmp_path, monkeypatch
):
    pgn = tmp_path / "games.pgn"
    output = tmp_path / "tiny.jsonl"
    pgn.write_text(PGN, encoding="utf-8")
    monkeypatch.setattr(
        cli.StockfishAnalyzer,
        "open",
        lambda *args, **kwargs: FakeAnalyzer(),
    )

    exit_code = cli.main(
        [
            "build",
            "--pgn",
            str(pgn),
            "--output",
            str(output),
            "--count",
            "6",
            "--minimum-ply",
            "0",
            "--max-per-game",
            "3",
            "--validation-fraction",
            "0.5",
            "--nodes",
            "100",
        ]
    )

    examples = load_training_examples(output)
    validate_game_isolation(examples)
    manifest = json.loads(
        output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert len(examples) == 6
    assert set(manifest["split_counts"]) == {"train", "validation"}
    assert manifest["engine"]["name"] == "Fixturefish"
