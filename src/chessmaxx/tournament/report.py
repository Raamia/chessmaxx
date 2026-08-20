"""Tournament capability summaries and calibrated-rating inputs."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Sequence

from chessmaxx.tournament.rating import RatingObservation, estimate_rating
from chessmaxx.tournament.schema import GameResult


def summarize_tournament(
    results: Sequence[GameResult],
    *,
    model_id: str,
    opponent_ratings: dict[str, float],
) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot summarize an empty tournament")
    by_opponent: dict[str, list[GameResult]] = {}
    observations: list[RatingObservation] = []
    scores: list[float] = []
    model_latencies: list[float] = []
    model_illegal_forfeits = 0
    model_moves = 0
    terminations = Counter(result.termination for result in results)
    for result in results:
        opponent_id = (
            result.black_id if result.white_id == model_id else result.white_id
        )
        if model_id not in {result.white_id, result.black_id}:
            raise ValueError("tournament result does not contain the model")
        by_opponent.setdefault(opponent_id, []).append(result)
        score = result.score_for(model_id)
        scores.append(score)
        if opponent_id in opponent_ratings:
            observations.append(
                RatingObservation(opponent_id, opponent_ratings[opponent_id], score)
            )
        for move in result.moves:
            if move.player_id == model_id:
                model_moves += 1
                model_latencies.append(move.latency_ms)
                if not move.legal:
                    model_illegal_forfeits += 1

    def summarize_group(games: Sequence[GameResult]) -> dict[str, Any]:
        group_scores = [game.score_for(model_id) for game in games]
        return {
            "games": len(games),
            "wins": sum(score == 1 for score in group_scores),
            "draws": sum(score == 0.5 for score in group_scores),
            "losses": sum(score == 0 for score in group_scores),
            "score_rate": sum(group_scores) / len(group_scores),
        }

    return {
        **summarize_group(results),
        "model_illegal_forfeits": model_illegal_forfeits,
        "model_move_attempts": model_moves,
        "model_first_try_legal_move_rate": (
            (model_moves - model_illegal_forfeits) / model_moves
            if model_moves
            else None
        ),
        "mean_game_plies": mean(len(result.moves) for result in results),
        "model_mean_move_latency_ms": (
            mean(model_latencies) if model_latencies else None
        ),
        "terminations": dict(sorted(terminations.items())),
        "by_opponent": {
            opponent: summarize_group(games)
            for opponent, games in sorted(by_opponent.items())
        },
        "by_model_color": {
            "white": summarize_group(
                [result for result in results if result.white_id == model_id]
            ),
            "black": summarize_group(
                [result for result in results if result.black_id == model_id]
            ),
        },
        "calibrated_elo": (
            estimate_rating(observations).to_dict() if observations else None
        ),
    }
