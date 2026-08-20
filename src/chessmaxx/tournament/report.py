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
    selection: str = "greedy",
) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot summarize an empty tournament")
    by_opponent: dict[str, list[GameResult]] = {}
    observations: list[RatingObservation] = []
    scores: list[float] = []
    model_latencies: list[float] = []
    model_illegal_forfeits = 0
    model_moves = 0
    model_attempts = 0
    first_attempt_legal = 0
    eventual_legal = 0
    moves_requiring_retry = 0
    successful_corrections = 0
    attempt_errors: Counter[str] = Counter()
    prompt_tokens = 0
    output_tokens = 0
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
                attempts = move.attempts
                attempt_count = len(attempts) if attempts else 1
                model_attempts += attempt_count
                first_legal = attempts[0].legal if attempts else move.legal
                first_attempt_legal += int(first_legal)
                eventual_legal += int(move.legal)
                if attempt_count > 1:
                    moves_requiring_retry += 1
                    successful_corrections += int(move.legal)
                for attempt in attempts:
                    if attempt.error is not None:
                        attempt_errors[attempt.error] += 1
                    prompt_tokens += attempt.prompt_tokens or 0
                    output_tokens += attempt.output_tokens or 0
                if not move.legal:
                    model_illegal_forfeits += 1

    def summarize_group(games: Sequence[GameResult]) -> dict[str, Any]:
        group_scores = [game.score_for(model_id) for game in games]
        return {
            "games": len(games),
            "wins": sum(score == 1 for score in group_scores),
            "draws": sum(score == 0.5 for score in group_scores),
            "losses": sum(score == 0 for score in group_scores),
            "score_rate": (
                sum(group_scores) / len(group_scores) if group_scores else None
            ),
        }

    generated_moves = selection != "legal-rerank"
    return {
        **summarize_group(results),
        "selection_mode": selection,
        "model_illegal_forfeits": model_illegal_forfeits,
        "model_plies": model_moves,
        "model_move_attempts": model_attempts if generated_moves else None,
        "model_selected_move_legal_rate": (
            eventual_legal / model_moves if model_moves else None
        ),
        "model_first_attempt_legal_rate": (
            first_attempt_legal / model_moves
            if generated_moves and model_moves
            else None
        ),
        "model_eventual_legal_move_rate": (
            eventual_legal / model_moves
            if generated_moves and model_moves
            else None
        ),
        "model_mean_attempts_per_move": (
            model_attempts / model_moves
            if generated_moves and model_moves
            else None
        ),
        "model_moves_requiring_retry": (
            moves_requiring_retry if generated_moves else None
        ),
        "model_successful_corrections": (
            successful_corrections if generated_moves else None
        ),
        "model_correction_success_rate": (
            successful_corrections / moves_requiring_retry
            if generated_moves and moves_requiring_retry
            else None
        ),
        "model_attempt_errors": (
            dict(sorted(attempt_errors.items())) if generated_moves else None
        ),
        "model_prompt_tokens": prompt_tokens if generated_moves else None,
        "model_output_tokens": output_tokens if generated_moves else None,
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
