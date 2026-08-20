"""Transparent Elo maximum-likelihood estimates with uncertainty bounds."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class RatingObservation:
    opponent_id: str
    opponent_rating: float
    score: float

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 1:
            raise ValueError("game score must be between zero and one")
        if not 0 < self.opponent_rating < 5000:
            raise ValueError("opponent rating must be between zero and 5000")


@dataclass(frozen=True, slots=True)
class RatingEstimate:
    status: str
    games: int
    wins: int
    draws: int
    losses: int
    score_rate: float
    rating: float | None
    confidence_lower: float | None
    confidence_upper: float | None
    confidence_level: float
    method: str = "logistic_mle_wilson_score_interval"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def expected_score(player_rating: float, opponent_rating: float) -> float:
    return 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))


def _rating_for_average_score(
    opponent_ratings: Sequence[float], target_score: float
) -> float | None:
    if target_score <= 0 or target_score >= 1:
        return None
    low = -5000.0
    high = 10000.0
    for _ in range(100):
        midpoint = (low + high) / 2
        predicted = sum(
            expected_score(midpoint, opponent) for opponent in opponent_ratings
        ) / len(opponent_ratings)
        if predicted < target_score:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def _wilson_interval(score_rate: float, games: int, z: float) -> tuple[float, float]:
    denominator = 1 + z * z / games
    center = (score_rate + z * z / (2 * games)) / denominator
    radius = (
        z
        * math.sqrt(
            score_rate * (1 - score_rate) / games
            + z * z / (4 * games * games)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def estimate_rating(
    observations: Sequence[RatingObservation],
    *,
    confidence_level: float = 0.95,
) -> RatingEstimate:
    """Fit one player against fixed Elo anchors; report censored sweeps honestly."""

    if not observations:
        raise ValueError("rating requires at least one calibrated game")
    if confidence_level != 0.95:
        raise ValueError("only the audited 95% confidence interval is supported")
    games = len(observations)
    wins = sum(observation.score == 1 for observation in observations)
    draws = sum(observation.score == 0.5 for observation in observations)
    losses = games - wins - draws
    score_rate = sum(observation.score for observation in observations) / games
    opponent_ratings = [observation.opponent_rating for observation in observations]
    lower_score, upper_score = _wilson_interval(score_rate, games, 1.959963984540054)
    lower_rating = _rating_for_average_score(opponent_ratings, lower_score)
    upper_rating = _rating_for_average_score(opponent_ratings, upper_score)
    if score_rate == 0:
        status = "below_ladder"
        rating = None
        lower_rating = None
    elif score_rate == 1:
        status = "above_ladder"
        rating = None
        upper_rating = None
    else:
        status = "estimated"
        rating = _rating_for_average_score(opponent_ratings, score_rate)
    return RatingEstimate(
        status=status,
        games=games,
        wins=wins,
        draws=draws,
        losses=losses,
        score_rate=score_rate,
        rating=rating,
        confidence_lower=lower_rating,
        confidence_upper=upper_rating,
        confidence_level=confidence_level,
    )
