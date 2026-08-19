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


def collect_model_identity(
    model: Any,
    tokenizer: Any,
    *,
    model_name: str,
    requested_revision: str,
    torch_version: str,
    transformers_version: str,
) -> dict[str, Any]:
    """Capture enough immutable model details to audit a baseline report."""

    config = model.config
    resolved_revision = getattr(config, "_commit_hash", None)
    if resolved_revision is None:
        resolved_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    trainable_count = sum(
        parameter.numel()
        for parameter in parameters
        if getattr(parameter, "requires_grad", False)
    )
    try:
        tokenizer_size = len(tokenizer)
    except TypeError:
        tokenizer_size = None

    return {
        "adapter": "huggingface",
        "model": model_name,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "dtype": str(getattr(model, "dtype", "unknown")).removeprefix("torch."),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "vocab_size": getattr(config, "vocab_size", None),
        "tokenizer_size": tokenizer_size,
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }


class HuggingFaceMoveGenerator:
    """Greedy batched generation for causal language models."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        device: str,
        max_new_tokens: int = 8,
        revision: str = "main",
        transformers_version: str = "unknown",
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
        self.revision = revision
        self.transformers_version = transformers_version
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
        revision: str = "main",
        dtype: str = "auto",
    ) -> "HuggingFaceMoveGenerator":
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face evaluation requires `pip install -e '.[model]'`"
            ) from exc

        selected_device = device
        if selected_device in (None, "auto"):
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            selected_dtype: str | Any = "auto"
        else:
            selected_dtype = getattr(torch, dtype, None)
            if selected_dtype is None:
                raise ValueError(f"unsupported torch dtype: {dtype}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=selected_dtype,
        ).to(selected_device)
        return cls(
            model,
            tokenizer,
            model_name,
            selected_device,
            max_new_tokens,
            revision,
            transformers.__version__,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        identity = collect_model_identity(
            self.model,
            self.tokenizer,
            model_name=self.model_name,
            requested_revision=self.revision,
            torch_version=self._torch.__version__,
            transformers_version=self.transformers_version,
        )
        return {
            **identity,
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
