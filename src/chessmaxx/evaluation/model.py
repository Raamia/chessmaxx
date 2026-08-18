"""Model-independent move generation and a Hugging Face implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol


def build_prompt(fen: str) -> str:
    """Create the canonical prompt used by every positional evaluation."""

    return (
        "Choose the best chess move for the position below. "
        "Respond with exactly one move in UCI notation and no explanation.\n"
        f"FEN: {fen}\n"
        "Move:"
    )


@dataclass(frozen=True, slots=True)
class GeneratedMove:
    """One raw model completion with timing and token counts."""

    raw_output: str
    latency_ms: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class MoveGenerator(Protocol):
    """Interface allowing local, remote, and test generators to share a runner."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    def generate_many(self, fens: list[str]) -> list[GeneratedMove]: ...


class HuggingFaceMoveGenerator:
    """Greedy batched generation for causal language models."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        device: str,
        max_new_tokens: int = 8,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        import torch

        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str | None = None,
        max_new_tokens: int = 8,
    ) -> "HuggingFaceMoveGenerator":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face evaluation requires `pip install -e '.[model]'`"
            ) from exc

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        ).to(selected_device)
        return cls(model, tokenizer, model_name, selected_device, max_new_tokens)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": "huggingface",
            "model": self.model_name,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "decoding": "greedy",
        }

    def generate_many(self, fens: list[str]) -> list[GeneratedMove]:
        if not fens:
            return []
        prompts = [build_prompt(fen) for fen in fens]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(self.device)
        input_width = inputs["input_ids"].shape[1]

        self._synchronize()
        started = time.perf_counter()
        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        self._synchronize()
        latency_per_position = (time.perf_counter() - started) * 1000 / len(fens)

        responses: list[GeneratedMove] = []
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for row, prompt_length in zip(generated, prompt_lengths, strict=True):
            completion = row[input_width:]
            raw_output = self.tokenizer.decode(
                completion, skip_special_tokens=True
            ).strip()
            responses.append(
                GeneratedMove(
                    raw_output=raw_output,
                    latency_ms=latency_per_position,
                    prompt_tokens=int(prompt_length),
                    output_tokens=int(completion.shape[0]),
                )
            )
        return responses

    def _synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()

