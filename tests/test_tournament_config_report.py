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
    assert assisted_profile.vocabulary_chunk_size == 4096


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
    assert summary["model_first_attempt_legal_rate"] == 0.0
    assert summary["model_selected_move_legal_rate"] == 0.0
    assert summary["by_model_color"]["white"]["games"] == 1
    assert summary["by_model_color"]["black"]["games"] == 1
    assert summary["calibrated_elo"]["games"] == 1
    assert summary["calibrated_elo"]["status"] == "below_ladder"
    json.dumps(summary)


def test_tournament_summary_distinguishes_correction_from_first_try_legality():
    schedule = ScheduledGame(
        game_id="assisted",
        opening_id="start",
        initial_fen=chess.STARTING_FEN,
        white_id="model",
        black_id="opponent",
        seed=1,
    )
    game = ActiveGame(
        schedule, assisted_player_id="model", max_attempts=3
    )
    assert game.apply(GeneratedMove("e2e5", latency_ms=1.0)) is None
    assert game.apply(GeneratedMove("e2e4", latency_ms=2.0)) is None
    result = game.apply(GeneratedMove("bad", latency_ms=1.0))
    assert result is not None

    summary = summarize_tournament(
        [result], model_id="model", opponent_ratings={}, selection="retry"
    )

    assert summary["model_plies"] == 1
    assert summary["model_move_attempts"] == 2
    assert summary["model_first_attempt_legal_rate"] == 0.0
    assert summary["model_eventual_legal_move_rate"] == 1.0
    assert summary["model_mean_attempts_per_move"] == 2.0
    assert summary["model_moves_requiring_retry"] == 1
    assert summary["model_successful_corrections"] == 1
    assert summary["model_correction_success_rate"] == 1.0
    assert summary["model_attempt_errors"] == {"illegal_move": 1}
    assert summary["by_model_color"]["black"]["score_rate"] is None

    reranked = summarize_tournament(
        [result], model_id="model", opponent_ratings={}, selection="legal-rerank"
    )
    assert reranked["model_selected_move_legal_rate"] == 1.0
    assert reranked["model_first_attempt_legal_rate"] is None
    assert reranked["model_move_attempts"] is None
