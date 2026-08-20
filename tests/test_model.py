import sys
from types import SimpleNamespace

import pytest

from chessmaxx.evaluation.model import (
    GenerationOOMError,
    HuggingFaceMoveGenerator,
    adaptive_batch_call,
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

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False


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


def test_loads_peft_adapter_over_explicit_base_model(monkeypatch, tmp_path):
    model = FakeModel()
    tokenizer = FakeTokenizer()
    tokenizer.padding_side = "right"
    calls = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            calls["tokenizer"] = path
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls["base"] = (name, kwargs)
            return model

    class PeftModel:
        @staticmethod
        def from_pretrained(base, path):
            calls["adapter"] = (base, path)
            return base

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float16="float16",
        __version__="2.test",
        version=SimpleNamespace(cuda=None),
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModel,
        __version__="5.test",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=PeftModel))

    generator = HuggingFaceMoveGenerator.from_adapter(
        tmp_path / "adapter",
        base_model_name="Qwen/base",
        revision="pinned",
        device="cpu",
        dtype="float16",
    )

    assert calls["base"] == (
        "Qwen/base",
        {"revision": "pinned", "torch_dtype": "float16"},
    )
    assert generator.metadata["adapter"] == "peft"
    assert generator.metadata["base_model"] == "Qwen/base"
    assert generator.metadata["adapter_path"] == str((tmp_path / "adapter").resolve())


def test_adaptive_batching_bisects_memory_failures_and_preserves_order():
    attempts = []
    recoveries = []

    def call(items):
        attempts.append(list(items))
        if len(items) > 1:
            raise RuntimeError("CUDA out of memory")
        return [items[0] * 10]

    outputs = adaptive_batch_call(
        [1, 2, 3, 4],
        call,
        lambda error: "out of memory" in str(error),
        lambda: recoveries.append(True),
    )

    assert outputs == [10, 20, 30, 40]
    assert attempts[0] == [1, 2, 3, 4]
    assert len(recoveries) == 3


def test_adaptive_batching_does_not_hide_non_memory_failures():
    def call(items):
        raise RuntimeError("model configuration is broken")

    with pytest.raises(RuntimeError, match="configuration is broken"):
        adaptive_batch_call([1, 2], call, lambda error: False, lambda: None)


def test_adaptive_batching_reports_single_position_memory_failure():
    def call(items):
        raise RuntimeError("out of memory")

    with pytest.raises(GenerationOOMError, match="single position"):
        adaptive_batch_call(
            [1], call, lambda error: "out of memory" in str(error), lambda: None
        )


def test_generator_builds_fen_prompts_but_can_execute_explicit_prompts():
    generator = object.__new__(HuggingFaceMoveGenerator)
    captured = []
    generator._generate_prompt_batch = lambda prompts: captured.extend(prompts) or [
        f"response-{index}" for index in range(len(prompts))
    ]
    generator._is_out_of_memory = lambda error: False
    generator._recover_memory = lambda: None

    assert generator.generate_many(["fen-one"]) == ["response-0"]
    assert captured == [build_prompt("fen-one")]

    captured.clear()
    assert generator.generate_prompts(["Correct the illegal move."]) == ["response-0"]
    assert captured == ["Correct the illegal move."]


def test_generator_rejects_empty_explicit_prompt():
    generator = object.__new__(HuggingFaceMoveGenerator)

    with pytest.raises(ValueError, match="must not be empty"):
        generator.generate_prompts(["  "])
