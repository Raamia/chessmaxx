import chess
import pytest

from chessmaxx.tournament.schedule import OpeningPosition, paired_schedule


def test_schedule_pairs_every_opening_with_reversed_model_colors():
    openings = (
        OpeningPosition("start", chess.STARTING_FEN),
        OpeningPosition(
            "e4-e5",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        ),
    )

    schedules = paired_schedule(
        model_id="model",
        opponent_ids=["random", "material"],
        openings=openings,
        games_per_opponent=4,
        seed=2026,
    )

    assert len(schedules) == 8
    for start in range(0, len(schedules), 2):
        white, black = schedules[start : start + 2]
        assert white.opening_id == black.opening_id
        assert white.initial_fen == black.initial_fen
        assert white.seed == black.seed
        assert white.white_id == black.black_id == "model"
        assert white.black_id == black.white_id


def test_schedule_is_deterministic():
    arguments = {
        "model_id": "model",
        "opponent_ids": ["random"],
        "openings": [OpeningPosition("start", chess.STARTING_FEN)],
        "games_per_opponent": 2,
        "seed": 7,
    }

    assert paired_schedule(**arguments) == paired_schedule(**arguments)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_schedule_requires_complete_color_pairs(count):
    with pytest.raises(ValueError, match="positive and even"):
        paired_schedule(
            model_id="model",
            opponent_ids=["random"],
            openings=[OpeningPosition("start", chess.STARTING_FEN)],
            games_per_opponent=count,
            seed=1,
        )
