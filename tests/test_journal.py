import pytest

from chessmaxx.evaluation.journal import JournalError, ResultJournal
from chessmaxx.evaluation.metrics import PositionResult


def fixture_result():
    return PositionResult(
        position_id="position-1",
        fen="fixture-fen",
        raw_output="e2e4",
        candidate="e2e4",
        parsed_move="e2e4",
        is_legal=True,
        error=None,
        teacher_moves=("e2e4",),
        best_score_cp=20,
        model_score_cp=20,
        centipawn_regret=0,
        latency_ms=1.5,
    )


def test_journal_round_trips_an_appended_result(tmp_path):
    journal = ResultJournal(tmp_path / "progress.jsonl", "run-a")
    assert journal.load_or_create() == {}

    journal.append(fixture_result())

    assert journal.load_or_create() == {"position-1": fixture_result()}


def test_journal_rejects_progress_from_another_run(tmp_path):
    path = tmp_path / "progress.jsonl"
    ResultJournal(path, "run-a").load_or_create()

    with pytest.raises(JournalError, match="different run"):
        ResultJournal(path, "run-b").load_or_create()


def test_journal_rejects_an_empty_existing_file(tmp_path):
    path = tmp_path / "progress.jsonl"
    path.touch()

    with pytest.raises(JournalError, match="missing journal manifest"):
        ResultJournal(path, "run-a").load_or_create()

