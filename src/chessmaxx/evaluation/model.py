"""Model-independent move generation and a Hugging Face implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class GenerationOOMError(RuntimeError):
    """Raised when even one position cannot fit in accelerator memory."""


def adaptive_batch_call(
    items: list[InputT],
    call: Callable[[list[InputT]], list[OutputT]],
    is_recoverable: Callable[[Exception], bool],
    recover: Callable[[], None],
) -> list[OutputT]:
    """Bisect a batch after memory errors while preserving input order."""

    if not items:
        return []
    try:
        outputs = call(items)
    except Exception as exc:
        if not is_recoverable(exc):
            raise
        recover()
        if len(items) == 1:
            raise GenerationOOMError(
                "generation ran out of memory with a single position"
            ) from exc
        midpoint = len(items) // 2
        return adaptive_batch_call(
            items[:midpoint], call, is_recoverable, recover
        ) + adaptive_batch_call(items[midpoint:], call, is_recoverable, recover)
    if len(outputs) != len(items):
        raise RuntimeError("generation returned a different number of outputs than inputs")
    return outputs


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

    @property
    def telemetry(self) -> dict[str, Any]: ...

    def reset_telemetry(self) -> None: ...

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
        identity_overrides: dict[str, Any] | None = None,
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
        self.identity_overrides = identity_overrides or {}
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()
        self.reset_telemetry()

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

    @classmethod
    def from_adapter(
        cls,
        adapter_path: str | Path,
        *,
        base_model_name: str,
        revision: str,
        device: str | None = None,
        max_new_tokens: int = 8,
        dtype: str = "auto",
    ) -> "HuggingFaceMoveGenerator":
        """Load a PEFT adapter over an explicitly pinned base model."""

        try:
            import peft
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                "adapter evaluation requires `pip install -e '.[train]'`"
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

        adapter = Path(adapter_path).resolve()
        tokenizer = transformers.AutoTokenizer.from_pretrained(adapter)
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_name,
            revision=revision,
            torch_dtype=selected_dtype,
        )
        model = peft.PeftModel.from_pretrained(base_model, adapter).to(selected_device)
        return cls(
            model,
            tokenizer,
            base_model_name,
            selected_device,
            max_new_tokens,
            revision,
            transformers.__version__,
            identity_overrides={
                "adapter": "peft",
                "adapter_path": str(adapter),
                "base_model": base_model_name,
            },
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
            **self.identity_overrides,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "decoding": "greedy",
        }

    def reset_telemetry(self) -> None:
        self._generation_seconds = 0.0
        self._positions_generated = 0
        self._prompt_tokens = 0
        self._output_tokens = 0
        self._batch_attempts: list[int] = []
        self._successful_batch_sizes: list[int] = []
        self._oom_retries = 0
        if self.device.startswith("cuda"):
            self._torch.cuda.reset_peak_memory_stats(self.device)

    @property
    def telemetry(self) -> dict[str, Any]:
        seconds = self._generation_seconds
        total_tokens = self._prompt_tokens + self._output_tokens
        value: dict[str, Any] = {
            "generation_seconds": seconds,
            "positions_generated": self._positions_generated,
            "prompt_tokens": self._prompt_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": total_tokens,
            "positions_per_second": self._positions_generated / seconds if seconds else 0.0,
            "tokens_per_second": total_tokens / seconds if seconds else 0.0,
            "output_tokens_per_second": self._output_tokens / seconds if seconds else 0.0,
            "batch_attempts": list(self._batch_attempts),
            "successful_batch_sizes": list(self._successful_batch_sizes),
            "oom_retries": self._oom_retries,
            "peak_allocated_vram_bytes": None,
            "peak_reserved_vram_bytes": None,
            "total_vram_bytes": None,
            "device_name": "CPU",
            "device_capability": None,
            "cuda_version": self._torch.version.cuda,
        }
        if self.device.startswith("cuda"):
            properties = self._torch.cuda.get_device_properties(self.device)
            capability = self._torch.cuda.get_device_capability(self.device)
            value.update(
                {
                    "peak_allocated_vram_bytes": self._torch.cuda.max_memory_allocated(
                        self.device
                    ),
                    "peak_reserved_vram_bytes": self._torch.cuda.max_memory_reserved(
                        self.device
                    ),
                    "total_vram_bytes": properties.total_memory,
                    "device_name": properties.name,
                    "device_capability": f"{capability[0]}.{capability[1]}",
                }
            )
        return value

    def generate_many(self, fens: list[str]) -> list[GeneratedMove]:
        return adaptive_batch_call(
            fens,
            self._generate_batch,
            self._is_out_of_memory,
            self._recover_memory,
        )

    def _generate_batch(self, fens: list[str]) -> list[GeneratedMove]:
        self._batch_attempts.append(len(fens))
        started = time.perf_counter()
        try:
            prompts = [build_prompt(fen) for fen in fens]
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            ).to(self.device)
            input_width = inputs["input_ids"].shape[1]

            self._synchronize()
            with self._torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            self._synchronize()
        except Exception:
            self._generation_seconds += time.perf_counter() - started
            raise
        elapsed = time.perf_counter() - started
        self._generation_seconds += elapsed
        latency_per_position = elapsed * 1000 / len(fens)

        responses: list[GeneratedMove] = []
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for row, prompt_length in zip(generated, prompt_lengths, strict=True):
            completion = row[input_width:]
            raw_output = self.tokenizer.decode(
                completion, skip_special_tokens=True
            ).strip()
            output_tokens = sum(
                int(token_id) != self.tokenizer.pad_token_id for token_id in completion
            )
            responses.append(
                GeneratedMove(
                    raw_output=raw_output,
                    latency_ms=latency_per_position,
                    prompt_tokens=int(prompt_length),
                    output_tokens=output_tokens,
                )
            )
        self._positions_generated += len(responses)
        self._prompt_tokens += sum(response.prompt_tokens or 0 for response in responses)
        self._output_tokens += sum(response.output_tokens or 0 for response in responses)
        self._successful_batch_sizes.append(len(responses))
        return responses

    def _synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()

    def _is_out_of_memory(self, error: Exception) -> bool:
        oom_type = getattr(self._torch, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(error, oom_type):
            return True
        return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()

    def _recover_memory(self) -> None:
        self._oom_retries += 1
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()
