import pytest

from chessmaxx.training.config import TinySFTProfile, load_tiny_sft_profile


def test_loads_checked_in_tiny_sft_profile():
    profile = load_tiny_sft_profile("configs/train/tiny-sft-qwen3-0.6b.toml")

    assert profile.model_id == "Qwen/Qwen3-0.6B-Base"
    assert profile.method == "lora"
    assert profile.max_examples == 100
    assert profile.packing is True


def test_rejects_unknown_training_setting():
    with pytest.raises(ValueError, match="unknown tiny-SFT setting"):
        TinySFTProfile.from_dict(
            {"name": "bad", "model_id": "model", "mystery_optimizer": True}
        )


@pytest.mark.parametrize(
    "values",
    [
        {"name": "bad", "model_id": "model", "method": "magic"},
        {"name": "bad", "model_id": "model", "max_examples": 0},
        {"name": "bad", "model_id": "model", "warmup_ratio": 1.0},
    ],
)
def test_rejects_unsafe_training_values(values):
    with pytest.raises(ValueError):
        TinySFTProfile.from_dict(values)
