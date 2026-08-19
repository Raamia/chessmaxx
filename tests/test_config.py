from pathlib import Path

import pytest

from chessmaxx.evaluation.config import BaselineProfile, load_baseline_profile


def test_loads_primary_baseline_profile():
    profile = load_baseline_profile("configs/baseline/qwen3-0.6b-base.toml")

    assert profile.name == "qwen3-0.6b-base"
    assert profile.model_id == "Qwen/Qwen3-0.6B-Base"
    assert profile.batch_size == 8


def test_rejects_unknown_settings(tmp_path: Path):
    path = tmp_path / "bad.toml"
    path.write_text(
        '[baseline]\nname = "bad"\nmodel_id = "model"\nsurprise = true\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown baseline setting"):
        load_baseline_profile(path)


@pytest.mark.parametrize("field", ["batch_size", "stockfish_nodes"])
def test_rejects_non_positive_work_settings(field):
    values = {"name": "bad", "model_id": "model", field: 0}

    with pytest.raises(ValueError, match=f"{field} must be positive"):
        BaselineProfile.from_dict(values)
