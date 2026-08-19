import chess
import pytest

from chessmaxx.evaluation.schema import EvaluationPosition, TeacherMove
from chessmaxx.training.label import LabelJournalError, TrainingLabeler


class FakeAnalyzer:
    engine_id = {"name": "Fixturefish"}

    def __init__(self):
        self.calls = 0

    def analyze_fen(self, fen):
        self.calls += 1
        return (TeacherMove("e2e4", 30), TeacherMove("d2d4", 20))


def positions():
    return [
        EvaluationPosition(
            position_id=f"game-1-ply-{ply}",
            game_id="game-1",
            ply=ply,
            fen=chess.STARTING_FEN,
            metadata={"source": "fixture.pgn"},
        )
        for ply in range(2)
    ]


def test_labeling_is_ordered_and_resumes_without_engine_calls(tmp_path):
    analyzer = FakeAnalyzer()
    labeler = TrainingLabeler(analyzer, {"game-1": "train"})
    journal = tmp_path / "labels.progress.jsonl"

    first = labeler.run(positions(), journal_path=journal)
    assert analyzer.calls == 2
    second = labeler.run(positions(), journal_path=journal)

    assert second == first
    assert analyzer.calls == 2
    assert [example.example_id for example in second] == [
        "game-1-ply-0",
        "game-1-ply-1",
    ]


def test_label_journal_rejects_changed_engine_identity(tmp_path):
    journal = tmp_path / "labels.progress.jsonl"
    TrainingLabeler(FakeAnalyzer(), {"game-1": "train"}).run(
        positions(), journal_path=journal
    )
    changed = FakeAnalyzer()
    changed.engine_id = {"name": "Different fish"}

    with pytest.raises(LabelJournalError, match="different run"):
        TrainingLabeler(changed, {"game-1": "train"}).run(
            positions(), journal_path=journal
        )


def test_labeling_requires_source_game_identity():
    position = EvaluationPosition("missing-game", chess.STARTING_FEN)

    with pytest.raises(ValueError, match="has no game_id"):
        TrainingLabeler(FakeAnalyzer(), {}).run([position])
