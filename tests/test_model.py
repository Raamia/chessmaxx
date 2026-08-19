import pytest

from chessmaxx.evaluation.model import HuggingFaceMoveGenerator, build_prompt


def test_prompt_has_fen_and_strict_output_instruction():
    fen = "8/8/8/8/8/8/6k1/4K3 w - - 0 1"

    prompt = build_prompt(fen)

    assert fen in prompt
    assert "exactly one move" in prompt
    assert prompt.endswith("Move:")


def test_generator_rejects_invalid_generation_length_before_model_use():
    with pytest.raises(ValueError, match="max_new_tokens"):
        HuggingFaceMoveGenerator(None, None, "fake", "cpu", max_new_tokens=0)

