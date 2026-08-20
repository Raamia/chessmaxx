"""Model-independent move generation and a Hugging Face implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

from chessmaxx.training.sparse_loss import chunked_target_log_probabilities


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class GenerationOOMError(RuntimeError):
    """Raised when even one position cannot fit in accelerator memory."""


def chunked_response_log_likelihoods(
    hidden_states: Any,
    input_ids: Any,
    response_starts: list[int],
    response_lengths: list[int],
    weight: Any,
    *,
    bias: Any | None = None,
    vocabulary_chunk_size: int,
) -> Any:
    """Score response spans exactly without materializing dense vocabulary logits."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("chunked response scoring requires PyTorch") from exc
    if hidden_states.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected hidden [B,L,H] and input IDs [B,L]")
    if hidden_states.shape[:2] != input_ids.shape:
        raise ValueError("hidden states and input IDs have different dimensions")
    rows = hidden_states.shape[0]
    if len(response_starts) != rows or len(response_lengths) != rows:
        raise ValueError("response spans must match the scoring batch")
    span_hidden: list[Any] = []
    span_targets: list[Any] = []
    for row, (start, length) in enumerate(
        zip(response_starts, response_lengths, strict=True)
    ):
        if start <= 0 or length <= 0 or start + length > input_ids.shape[1]:
            raise ValueError("response span is outside the candidate sequence")
        span_hidden.append(hidden_states[row, start - 1 : start + length - 1])
        span_targets.append(input_ids[row, start : start + length])
    flat_hidden = torch.cat(span_hidden, dim=0)
    flat_targets = torch.cat(span_targets, dim=0)
    token_scores = chunked_target_log_probabilities(
        flat_hidden,
        weight.detach(),
        flat_targets,
        bias=bias.detach() if bias is not None else None,
        chunk_size=vocabulary_chunk_size,
    )
    return torch.stack(
        [score.sum() for score in token_scores.split(response_lengths)]
    )


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


class PromptMoveGenerator(MoveGenerator, Protocol):
    """A move generator that can execute explicit feedback-aware prompts."""

    def generate_prompts(self, prompts: list[str]) -> list[GeneratedMove]: ...


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
        return self.generate_prompts([build_prompt(fen) for fen in fens])

    def generate_prompts(self, prompts: list[str]) -> list[GeneratedMove]:
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("generation prompts must not be empty")
        return adaptive_batch_call(
            prompts,
            self._generate_prompt_batch,
            self._is_out_of_memory,
            self._recover_memory,
        )

    def _generate_prompt_batch(self, prompts: list[str]) -> list[GeneratedMove]:
        self._batch_attempts.append(len(prompts))
        started = time.perf_counter()
        try:
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
        latency_per_position = elapsed * 1000 / len(prompts)

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


class HuggingFaceLegalMoveRanker(HuggingFaceMoveGenerator):
    """Choose the highest-likelihood move from the board's legal move set."""

    def __init__(self, *args: Any, candidate_batch_size: int = 16, **kwargs: Any) -> None:
        if candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be positive")
        self.candidate_batch_size = candidate_batch_size
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str | None = None,
        revision: str = "main",
        dtype: str = "auto",
        candidate_batch_size: int = 16,
    ) -> "HuggingFaceLegalMoveRanker":
        loaded = HuggingFaceMoveGenerator.from_pretrained(
            model_name,
            device=device,
            revision=revision,
            dtype=dtype,
            max_new_tokens=8,
        )
        return cls(
            loaded.model,
            loaded.tokenizer,
            loaded.model_name,
            loaded.device,
            max_new_tokens=8,
            revision=loaded.revision,
            transformers_version=loaded.transformers_version,
            identity_overrides={
                **loaded.identity_overrides,
                "selection_mode": "legal_move_likelihood_rerank",
            },
            candidate_batch_size=candidate_batch_size,
        )

    @classmethod
    def from_adapter(
        cls,
        adapter_path: str | Path,
        *,
        base_model_name: str,
        revision: str,
        device: str | None = None,
        dtype: str = "auto",
        candidate_batch_size: int = 16,
    ) -> "HuggingFaceLegalMoveRanker":
        loaded = HuggingFaceMoveGenerator.from_adapter(
            adapter_path,
            base_model_name=base_model_name,
            revision=revision,
            device=device,
            dtype=dtype,
            max_new_tokens=8,
        )
        return cls(
            loaded.model,
            loaded.tokenizer,
            loaded.model_name,
            loaded.device,
            max_new_tokens=8,
            revision=loaded.revision,
            transformers_version=loaded.transformers_version,
            identity_overrides={
                **loaded.identity_overrides,
                "selection_mode": "legal_move_likelihood_rerank",
            },
            candidate_batch_size=candidate_batch_size,
        )

    def reset_telemetry(self) -> None:
        super().reset_telemetry()
        self._candidate_sequences_scored = 0
        self._candidate_input_tokens = 0
        self._candidate_batch_attempts: list[int] = []

    @property
    def telemetry(self) -> dict[str, Any]:
        return {
            **super().telemetry,
            "candidate_sequences_scored": self._candidate_sequences_scored,
            "candidate_input_tokens": self._candidate_input_tokens,
            "candidate_batch_attempts": list(self._candidate_batch_attempts),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata,
            "candidate_batch_size": self.candidate_batch_size,
        }

    def generate_many(self, fens: list[str]) -> list[GeneratedMove]:
        return [self._rank_legal_moves(fen) for fen in fens]

    def _rank_legal_moves(self, fen: str) -> GeneratedMove:
        import chess

        board = chess.Board(fen)
        moves = sorted(move.uci() for move in board.legal_moves)
        if not moves:
            raise ValueError("cannot rank moves for a terminal position")
        started = time.perf_counter()
        scores: list[float] = []
        selected_output_tokens = 0
        prompt_ids = self.tokenizer.encode(build_prompt(fen), add_special_tokens=True)
        for start in range(0, len(moves), self.candidate_batch_size):
            candidates = moves[start : start + self.candidate_batch_size]
            self._candidate_batch_attempts.append(len(candidates))
            rows: list[list[int]] = []
            response_starts: list[int] = []
            response_lengths: list[int] = []
            for move in candidates:
                target_ids = self.tokenizer.encode(
                    f" {move}", add_special_tokens=False
                ) + [self.tokenizer.eos_token_id]
                rows.append([*prompt_ids, *target_ids])
                response_starts.append(len(prompt_ids))
                response_lengths.append(len(target_ids))
            width = max(len(row) for row in rows)
            input_ids = [
                row + [self.tokenizer.pad_token_id] * (width - len(row))
                for row in rows
            ]
            attention_mask = [
                [1] * len(row) + [0] * (width - len(row)) for row in rows
            ]
            input_tensor = self._torch.tensor(
                input_ids, dtype=self._torch.long, device=self.device
            )
            attention_tensor = self._torch.tensor(
                attention_mask, dtype=self._torch.long, device=self.device
            )
            self._synchronize()
            with self._torch.inference_mode():
                logits = self.model(
                    input_ids=input_tensor,
                    attention_mask=attention_tensor,
                    use_cache=False,
                ).logits
                log_probabilities = self._torch.nn.functional.log_softmax(
                    logits[:, :-1, :].float(), dim=-1
                )
            self._synchronize()
            for row_index, (response_start, response_length) in enumerate(
                zip(response_starts, response_lengths, strict=True)
            ):
                targets = input_tensor[
                    row_index,
                    response_start : response_start + response_length,
                ]
                token_scores = log_probabilities[
                    row_index,
                    response_start - 1 : response_start + response_length - 1,
                ].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                scores.append(float(token_scores.sum().item()))
            self._candidate_sequences_scored += len(candidates)
            self._candidate_input_tokens += sum(map(sum, attention_mask))
        best_index = max(range(len(moves)), key=lambda index: scores[index])
        selected_output_tokens = len(
            self.tokenizer.encode(f" {moves[best_index]}", add_special_tokens=False)
        )
        elapsed = time.perf_counter() - started
        self._generation_seconds += elapsed
        self._positions_generated += 1
        self._prompt_tokens += len(prompt_ids)
        self._output_tokens += selected_output_tokens
        self._batch_attempts.append(1)
        self._successful_batch_sizes.append(1)
        return GeneratedMove(
            raw_output=moves[best_index],
            latency_ms=elapsed * 1000,
            prompt_tokens=len(prompt_ids),
            output_tokens=selected_output_tokens,
        )
