import json

import pytest

from chessmaxx.tournament.compare import compare_reports, main


def report(source, *, selection="retry", game_id="game-1", score=0.0):
    return {
        "schema_version": 1,
        "settings": {
            "model_source": source,
            "adapter_sha256": "adapter" if source == "adapter" else None,
            "selection": selection,
            "context": "fen",
            "max_attempts": 3,
            "profile": {"model_player_id": "model"},
        },
        "players": {"model": {"model": "Qwen/base", "source": source}},
        "summary": {
            "games": 1,
            "wins": int(score == 1),
            "draws": int(score == 0.5),
            "losses": int(score == 0),
            "score_rate": score,
            "model_first_attempt_legal_rate": score,
            "model_eventual_legal_move_rate": score,
            "model_selected_move_legal_rate": score,
            "model_mean_attempts_per_move": 2.0,
            "model_illegal_forfeits": int(score == 0),
            "calibrated_elo": None,
        },
        "games": [
            {
                "game_id": game_id,
                "opening_id": "start",
                "initial_fen": "fen",
                "white_id": "model",
                "black_id": "opponent",
            }
        ],
    }


def write_report(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_comparison_calculates_adapter_delta_on_matching_schedule(tmp_path):
    base = write_report(tmp_path / "base.json", report("base", score=0.0))
    adapter = write_report(
        tmp_path / "adapter.json", report("adapter", score=0.5)
    )

    comparison = compare_reports([base, adapter])

    assert len(comparison["conditions"]) == 2
    assert comparison["adapter_minus_base"] == [
        {
            "selection": "retry",
            "context": "fen",
            "max_attempts": 3,
            "score_rate_delta": 0.5,
            "first_attempt_legal_rate_delta": 0.5,
            "eventual_legal_rate_delta": 0.5,
        }
    ]


def test_comparison_rejects_different_schedules(tmp_path):
    first = write_report(tmp_path / "one.json", report("base"))
    second = write_report(
        tmp_path / "two.json", report("adapter", game_id="different")
    )

    with pytest.raises(ValueError, match="same game schedule"):
        compare_reports([first, second])


def test_comparison_cli_writes_json(tmp_path):
    base = write_report(tmp_path / "base.json", report("base"))
    adapter = write_report(tmp_path / "adapter.json", report("adapter"))
    output = tmp_path / "comparison.json"

    assert main(
        ["--reports", str(base), str(adapter), "--output", str(output)]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
