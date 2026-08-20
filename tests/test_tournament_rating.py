import pytest

from chessmaxx.tournament.rating import (
    RatingObservation,
    estimate_rating,
    expected_score,
)


def observations(scores, rating=1500):
    return [
        RatingObservation("anchor", opponent_rating=rating, score=score)
        for score in scores
    ]


def test_equal_score_estimates_the_anchor_rating():
    estimate = estimate_rating(observations([1, 0] * 20))

    assert estimate.status == "estimated"
    assert estimate.rating == pytest.approx(1500)
    assert estimate.wins == 20
    assert estimate.losses == 20
    assert estimate.confidence_lower < estimate.rating < estimate.confidence_upper


def test_seventy_five_percent_score_matches_elo_logistic_scale():
    estimate = estimate_rating(observations([1, 1, 1, 0] * 20))

    assert estimate.rating == pytest.approx(1690.8485, abs=1e-3)
    assert expected_score(estimate.rating, 1500) == pytest.approx(0.75)


def test_all_losses_are_reported_as_below_ladder_not_fake_elo():
    estimate = estimate_rating(observations([0] * 40, rating=1320))

    assert estimate.status == "below_ladder"
    assert estimate.rating is None
    assert estimate.confidence_lower is None
    assert estimate.confidence_upper < 1320


def test_rating_requires_calibrated_games():
    with pytest.raises(ValueError, match="at least one"):
        estimate_rating([])
