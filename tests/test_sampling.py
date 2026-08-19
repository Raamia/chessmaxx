from collections import Counter

import chess

from chessmaxx.evaluation.sampling import classify_phase, sample_pgn_positions


PGN = """[Event "Game One"]
[Site "Test"]
[Date "2026.01.01"]
[Round "1"]
[White "Alpha"]
[Black "Beta"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O *

[Event "Game Two"]
[Site "Test"]
[Date "2026.01.01"]
[Round "2"]
[White "Gamma"]
[Black "Delta"]
[Result "*"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Rc1 c6 8. Bd3 dxc4 *
"""


def test_phase_classification_uses_ply_and_material():
    assert classify_phase(chess.Board(), ply=0) == "opening"
    assert classify_phase(chess.Board(), ply=30) == "middlegame"
    assert (
        classify_phase(chess.Board("8/8/8/4k3/8/3K4/4P3/8 w - - 0 1"), ply=40)
        == "endgame"
    )


def test_sampling_is_deterministic_and_limits_each_source_game(tmp_path):
    path = tmp_path / "games.pgn"
    path.write_text(PGN, encoding="utf-8")

    first = sample_pgn_positions(path, 5, seed=42, minimum_ply=0, max_per_game=3)
    second = sample_pgn_positions(path, 5, seed=42, minimum_ply=0, max_per_game=3)

    assert first == second
    assert len(first) == 5
    assert max(Counter(position.game_id for position in first).values()) <= 3
    assert len({position.position_id for position in first}) == 5


def test_different_seeds_change_the_sample(tmp_path):
    path = tmp_path / "games.pgn"
    path.write_text(PGN, encoding="utf-8")

    first = sample_pgn_positions(path, 4, seed=1, minimum_ply=0, max_per_game=4)
    second = sample_pgn_positions(path, 4, seed=2, minimum_ply=0, max_per_game=4)

    assert [position.position_id for position in first] != [
        position.position_id for position in second
    ]
