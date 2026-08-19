import json

import chess
import pytest

from chessmaxx.evaluation.dataset import DatasetError, load_positions, write_positions
from chessmaxx.evaluation.schema import EvaluationPosition, TeacherMove


def test_positions_round_trip(tmp_path):
    path = tmp_path / "positions.jsonl"
    expected = EvaluationPosition(
        position_id="game-1-ply-0",
        game_id="game-1",
        ply=0,
        phase="opening",
        fen=chess.STARTING_FEN,
        teacher_moves=(TeacherMove(move="e2e4", score_cp=31),),
        metadata={"source": "fixture"},
    )

    write_positions(path, [expected])

    assert load_positions(path) == [expected]


def test_loader_rejects_duplicate_position_ids(tmp_path):
    path = tmp_path / "positions.jsonl"
    row = {"position_id": "duplicate", "fen": chess.STARTING_FEN}
    path.write_text(f"{json.dumps(row)}\n{json.dumps(row)}\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="duplicate position_id"):
        load_positions(path)


def test_loader_reports_invalid_fen_with_line_number(tmp_path):
    path = tmp_path / "positions.jsonl"
    path.write_text(
        json.dumps({"position_id": "bad", "fen": "not a FEN"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match=r"positions.jsonl:1"):
        load_positions(path)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {"position_id": "terminal", "fen": "7k/7Q/7K/8/8/8/8/8 b - - 0 1"},
            "already terminal",
        ),
        (
            {
                "position_id": "illegal-teacher",
                "fen": chess.STARTING_FEN,
                "teacher_moves": [{"move": "e2e5", "score_cp": 10}],
            },
            "teacher move 'e2e5' is illegal",
        ),
        (
            {
                "position_id": "unordered",
                "fen": chess.STARTING_FEN,
                "teacher_moves": [
                    {"move": "e2e4", "score_cp": 10},
                    {"move": "d2d4", "score_cp": 20},
                ],
            },
            "descending score",
        ),
    ],
)
def test_loader_rejects_positions_that_break_evaluation_contract(
    tmp_path, row, message
):
    path = tmp_path / "positions.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(DatasetError, match=message):
        load_positions(path)

