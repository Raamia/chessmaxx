import chess
import pytest

from chessmaxx.evaluation.model import GeneratedMove
from chessmaxx.tournament.game import ActiveGame
from chessmaxx.tournament.journal import (
    TournamentJournal,
    TournamentJournalError,
    tournament_run_key,
)
from chessmaxx.tournament.schema import ScheduledGame


def schedule() -> ScheduledGame:
    return ScheduledGame(
        game_id="game-1",
        opening_id="start",
        initial_fen=chess.STARTING_FEN,
        white_id="model",
        black_id="random",
        seed=7,
    )


def result():
    game = ActiveGame(schedule())
    completed = game.apply(GeneratedMove("illegal", latency_ms=1.0))
    assert completed is not None
    return completed


def run_key():
    return tournament_run_key(
        [schedule()],
        players={"model": {"kind": "model"}, "random": {"kind": "random"}},
        settings={"batch_size": 8},
    )


def test_tournament_journal_round_trips_completed_game(tmp_path):
    journal = TournamentJournal(tmp_path / "games.jsonl", run_key())
    assert journal.load_or_create() == {}

    journal.append(result())

    assert journal.load_or_create() == {"game-1": result()}


def test_tournament_journal_rejects_changed_manifest(tmp_path):
    path = tmp_path / "games.jsonl"
    TournamentJournal(path, run_key()).load_or_create()

    with pytest.raises(TournamentJournalError, match="different tournament"):
        TournamentJournal(path, "another-run").load_or_create()


def test_tournament_run_key_changes_with_player_identity():
    first = tournament_run_key(
        [schedule()], players={"model": {"adapter": "a"}}, settings={}
    )
    second = tournament_run_key(
        [schedule()], players={"model": {"adapter": "b"}}, settings={}
    )

    assert first != second
