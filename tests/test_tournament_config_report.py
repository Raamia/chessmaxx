import json
from dataclasses import replace

import chess

from chessmaxx.evaluation.model import GeneratedMove
from chessmaxx.tournament.config import load_elo_profile, load_openings
from chessmaxx.tournament.game import ActiveGame
from chessmaxx.tournament.report import summarize_tournament
from chessmaxx.tournament.schema import ScheduledGame


def test_checked_in_elo_profile_has_balanced_ladder():
    profile = load_elo_profile("configs/elo/qwen3-0.6b-elo.toml")
    openings = load_openings("data/elo/openings-v1.jsonl")

    assert profile.games_per_opponent == 20
    assert profile.selection == "legal-rerank"
    assert [opponent.player_id for opponent in profile.opponents] == [
        "random",
        "material",
        "stockfish-1320",
    ]
    assert profile.opponents[-1].rating == 1320
    assert len(openings) == 8

    assisted = replace(
        profile, selection="retry", max_attempts=3, context="fen-pgn"
    )
    assert assisted.max_attempts == 3
    assert assisted.context == "fen-pgn"

    assisted_profile = load_elo_profile(
        "configs/elo/qwen3-0.6b-assisted.toml"
    )
    assert assisted_profile.selection == "retry"
    assert assisted_profile.max_attempts == 3
    assert assisted_profile.candidate_batch_size == 8


def completed_game(game_id, opponent, model_white, model_move):
    schedule = ScheduledGame(
        game_id=game_id,
        opening_id="start",
        initial_fen=chess.STARTING_FEN,
        white_id="model" if model_white else opponent,
        black_id=opponent if model_white else "model",
        seed=1,
    )
    game = ActiveGame(schedule)
    if model_white:
        result = game.apply(GeneratedMove(model_move, latency_ms=2.0))
    else:
        assert game.apply(GeneratedMove("e2e4", latency_ms=1.0)) is None
        result = game.apply(GeneratedMove(model_move, latency_ms=2.0))
    assert result is not None
    return result


def test_tournament_summary_separates_unrated_games_from_elo():
    results = [
        completed_game("g1", "random", True, "bad"),
        completed_game("g2", "sf", False, "bad"),
    ]

    summary = summarize_tournament(
        results, model_id="model", opponent_ratings={"sf": 1320}
    )

    assert summary["games"] == 2
    assert summary["model_illegal_forfeits"] == 2
    assert summary["model_move_attempts"] == 2
    assert summary["model_first_try_legal_move_rate"] == 0.0
    assert summary["by_model_color"]["white"]["games"] == 1
    assert summary["by_model_color"]["black"]["games"] == 1
    assert summary["calibrated_elo"]["games"] == 1
    assert summary["calibrated_elo"]["status"] == "below_ladder"
    json.dumps(summary)
