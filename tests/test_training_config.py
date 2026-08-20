from dataclasses import asdict, replace

import pytest

from chessmaxx.training.config import TinySFTProfile, load_tiny_sft_profile


def test_loads_checked_in_tiny_sft_profile():
    profile = load_tiny_sft_profile("configs/train/tiny-sft-qwen3-0.6b.toml")

    assert profile.model_id == "Qwen/Qwen3-0.6B-Base"
    assert profile.revision == "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    assert profile.method == "lora"
    assert profile.max_examples == 100
    assert profile.packing is True


def test_unpacked_control_only_changes_name_and_packing():
    packed = asdict(
        load_tiny_sft_profile("configs/train/tiny-sft-qwen3-0.6b.toml")
    )
    unpacked = asdict(
        load_tiny_sft_profile(
            "configs/train/tiny-sft-qwen3-0.6b-unpacked.toml"
        )
    )

    differences = {
        key: (packed[key], unpacked[key])
        for key in packed
        if packed[key] != unpacked[key]
    }

    assert differences == {
        "name": ("tiny-sft-qwen3-0.6b", "tiny-sft-qwen3-0.6b-unpacked"),
        "packing": (True, False),
    }


def test_isolated_profile_only_enables_attention_isolation():
    naive = asdict(
        load_tiny_sft_profile("configs/train/tiny-sft-qwen3-0.6b.toml")
    )
    isolated = asdict(
        load_tiny_sft_profile(
            "configs/train/tiny-sft-qwen3-0.6b-isolated.toml"
        )
    )

    differences = {
        key: (naive[key], isolated[key])
        for key in naive
        if naive[key] != isolated[key]
    }

    assert differences == {
        "name": (
            "tiny-sft-qwen3-0.6b",
            "tiny-sft-qwen3-0.6b-isolated",
        ),
        "isolate_packed_attention": (False, True),
    }


def test_scaled_profile_enables_held_out_training_evaluation():
    profile = load_tiny_sft_profile(
        "configs/train/scaled-sft-qwen3-0.6b-isolated.toml"
    )

    assert profile.max_examples == 900
    assert profile.max_validation_examples == 100
    assert profile.epochs == 5.0
    assert profile.packing is True
    assert profile.isolate_packed_attention is True
    assert profile.evaluation_steps == 25
    assert profile.save_total_limit == 20


def test_scaled_distillation_profile_is_a_dense_policy_control():
    profile = load_tiny_sft_profile(
        "configs/train/scaled-distill-qwen3-0.6b.toml"
    )

    assert profile.objective == "multipv_policy"
    assert profile.max_examples == 900
    assert profile.max_validation_examples == 100
    assert profile.packing is False
    assert profile.max_teacher_candidates == 3
    assert profile.teacher_temperature_cp == 100.0
    assert profile.hard_loss_weight == 0.5
    assert profile.policy_loss_backend == "dense"
    assert profile.vocabulary_chunk_size == 4096


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


def test_isolated_attention_requires_packing():
    profile = load_tiny_sft_profile(
        "configs/train/tiny-sft-qwen3-0.6b-unpacked.toml"
    )

    with pytest.raises(ValueError, match="requires packing"):
        replace(profile, isolate_packed_attention=True)


def test_policy_objective_rejects_sequence_packing():
    profile = load_tiny_sft_profile(
        "configs/train/tiny-sft-qwen3-0.6b-unpacked.toml"
    )

    with pytest.raises(ValueError, match="does not support packing"):
        replace(profile, objective="multipv_policy", packing=True)


def test_chunked_policy_backend_requires_policy_objective():
    profile = load_tiny_sft_profile(
        "configs/train/tiny-sft-qwen3-0.6b-unpacked.toml"
    )

    with pytest.raises(ValueError, match="multi-PV policy objective"):
        replace(profile, policy_loss_backend="chunked_exact")


@pytest.mark.parametrize("backend", ["sparse", "triton"])
def test_rejects_unknown_policy_loss_backend(backend):
    with pytest.raises(ValueError, match="policy_loss_backend"):
        TinySFTProfile(
            name="bad",
            model_id="model",
            objective="multipv_policy",
            packing=False,
            policy_loss_backend=backend,
        )


def test_chunked_policy_backend_requires_frozen_lora_output_head():
    with pytest.raises(ValueError, match="frozen output head"):
        TinySFTProfile(
            name="bad",
            model_id="model",
            method="full",
            objective="multipv_policy",
            packing=False,
            policy_loss_backend="chunked_exact",
        )
