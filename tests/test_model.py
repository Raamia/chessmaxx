import pytest

from chessmaxx.evaluation.model import (
    HuggingFaceMoveGenerator,
    build_prompt,
    collect_model_identity,
)


class FakeParameter:
    def __init__(self, size, requires_grad):
        self.size = size
        self.requires_grad = requires_grad

    def numel(self):
        return self.size


class FakeConfig:
    _commit_hash = "abc123"
    vocab_size = 151_936
    hidden_size = 1_024
    num_hidden_layers = 28


class FakeModel:
    config = FakeConfig()
    dtype = "torch.float16"

    def parameters(self):
        return [FakeParameter(100, True), FakeParameter(50, False)]


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def __len__(self):
        return 151_936


def test_prompt_has_fen_and_strict_output_instruction():
    fen = "8/8/8/8/8/8/6k1/4K3 w - - 0 1"

    prompt = build_prompt(fen)

    assert fen in prompt
    assert "exactly one move" in prompt
    assert prompt.endswith("Move:")


def test_generator_rejects_invalid_generation_length_before_model_use():
    with pytest.raises(ValueError, match="max_new_tokens"):
        HuggingFaceMoveGenerator(None, None, "fake", "cpu", max_new_tokens=0)


def test_collects_resolved_model_and_tokenizer_identity():
    identity = collect_model_identity(
        FakeModel(),
        FakeTokenizer(),
        model_name="Qwen/Qwen3-0.6B-Base",
        requested_revision="main",
        torch_version="2.test",
        transformers_version="4.test",
    )

    assert identity["resolved_revision"] == "abc123"
    assert identity["parameter_count"] == 150
    assert identity["trainable_parameter_count"] == 100
    assert identity["vocab_size"] == 151_936
    assert identity["dtype"] == "float16"
