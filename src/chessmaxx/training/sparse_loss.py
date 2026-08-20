"""Exact policy loss primitives that avoid dense sequence logits."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True, slots=True)
class CausalLMProjection:
    hidden_states: Any
    weight: Any
    bias: Any | None


def _validate_projection_inputs(
    hidden_states: Any,
    weight: Any,
    targets: Any,
    bias: Any | None,
    chunk_size: int,
) -> None:
    if hidden_states.ndim != 2 or weight.ndim != 2 or targets.ndim != 1:
        raise ValueError("expected hidden [N,H], weight [V,H], and targets [N]")
    if hidden_states.shape[0] != targets.shape[0]:
        raise ValueError("hidden states and targets have different token counts")
    if hidden_states.shape[1] != weight.shape[1]:
        raise ValueError("hidden states and vocabulary weights have different widths")
    if weight.shape[0] <= 0:
        raise ValueError("vocabulary projection must not be empty")
    if bias is not None and bias.shape != weight.shape[:1]:
        raise ValueError("vocabulary bias has the wrong shape")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if targets.numel() and (
        int(targets.min().item()) < 0
        or int(targets.max().item()) >= weight.shape[0]
    ):
        raise ValueError("target token is outside the vocabulary")


def chunked_target_log_probabilities(
    hidden_states: Any,
    weight: Any,
    targets: Any,
    *,
    bias: Any | None = None,
    chunk_size: int,
) -> Any:
    """Calculate exact target log probabilities without dense vocabulary logits."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("chunked exact loss requires PyTorch") from exc
    _validate_projection_inputs(hidden_states, weight, targets, bias, chunk_size)
    if weight.requires_grad or (bias is not None and bias.requires_grad):
        raise ValueError(
            "chunked exact loss requires a frozen vocabulary projection"
        )
    function = _chunked_target_log_probability_function()
    return function.apply(hidden_states, weight, targets, bias, chunk_size)


def chunked_candidate_sequence_log_likelihoods(
    hidden_states: Any,
    labels: Any,
    weight: Any,
    *,
    bias: Any | None = None,
    chunk_size: int,
) -> tuple[Any, Any]:
    """Sum exact response-token log probabilities for grouped candidates."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("chunked exact loss requires PyTorch") from exc
    if hidden_states.ndim != 4 or labels.ndim != 3:
        raise ValueError("expected hidden [B,K,L,H] and labels [B,K,L]")
    if hidden_states.shape[:3] != labels.shape:
        raise ValueError("hidden-state and label dimensions do not match")
    shifted_hidden = hidden_states[..., :-1, :]
    shifted_labels = labels[..., 1:]
    supervised = shifted_labels != -100
    supervised_hidden = shifted_hidden[supervised]
    targets = shifted_labels[supervised]
    token_log_probabilities = chunked_target_log_probabilities(
        supervised_hidden,
        weight,
        targets,
        bias=bias,
        chunk_size=chunk_size,
    )
    positioned_scores = torch.zeros_like(shifted_labels, dtype=torch.float32)
    positioned_scores = positioned_scores.masked_scatter(
        supervised, token_log_probabilities
    )
    return positioned_scores.sum(dim=-1), supervised.sum(dim=-1)


def _chunked_target_log_probability_forward(
    hidden_states: Any,
    weight: Any,
    targets: Any,
    bias: Any | None,
    chunk_size: int,
) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as functional

    log_normalizer = None
    for start in range(0, weight.shape[0], chunk_size):
        stop = min(start + chunk_size, weight.shape[0])
        chunk_bias = bias[start:stop] if bias is not None else None
        chunk_logits = functional.linear(
            hidden_states, weight[start:stop], chunk_bias
        ).float()
        chunk_normalizer = torch.logsumexp(chunk_logits, dim=-1)
        log_normalizer = (
            chunk_normalizer
            if log_normalizer is None
            else torch.logaddexp(log_normalizer, chunk_normalizer)
        )
    target_weights = weight.index_select(0, targets)
    target_logits = (hidden_states * target_weights).sum(dim=-1).float()
    if bias is not None:
        target_logits = target_logits + bias.index_select(0, targets).float()
    return target_logits - log_normalizer, log_normalizer


@lru_cache(maxsize=1)
def _chunked_target_log_probability_function() -> Any:
    import torch
    import torch.nn.functional as functional

    class ChunkedTargetLogProbability(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            hidden_states: Any,
            weight: Any,
            targets: Any,
            bias: Any | None,
            chunk_size: int,
        ) -> Any:
            log_probabilities, log_normalizer = (
                _chunked_target_log_probability_forward(
                    hidden_states,
                    weight,
                    targets,
                    bias,
                    chunk_size,
                )
            )
            saved_bias = (
                bias
                if bias is not None
                else hidden_states.new_empty((0,))
            )
            ctx.save_for_backward(
                hidden_states,
                weight,
                targets,
                log_normalizer,
                saved_bias,
            )
            ctx.has_bias = bias is not None
            ctx.chunk_size = chunk_size
            return log_probabilities

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
            hidden_states, weight, targets, log_normalizer, saved_bias = (
                ctx.saved_tensors
            )
            bias = saved_bias if ctx.has_bias else None
            expected_weight = torch.zeros_like(
                hidden_states, dtype=torch.float32
            )
            for start in range(0, weight.shape[0], ctx.chunk_size):
                stop = min(start + ctx.chunk_size, weight.shape[0])
                chunk_bias = bias[start:stop] if bias is not None else None
                chunk_logits = functional.linear(
                    hidden_states,
                    weight[start:stop],
                    chunk_bias,
                ).float()
                probabilities = torch.exp(
                    chunk_logits - log_normalizer.unsqueeze(-1)
                )
                expected_weight.add_(
                    probabilities @ weight[start:stop].float()
                )
            target_weight = weight.index_select(0, targets).float()
            grad_hidden = (
                grad_output.float().unsqueeze(-1)
                * (target_weight - expected_weight)
            )
            return grad_hidden.to(hidden_states.dtype), None, None, None, None

    return ChunkedTargetLogProbability


def causal_hidden_and_projection(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
) -> CausalLMProjection:
    """Run the transformer body without materializing vocabulary logits."""

    causal_model = (
        model.get_base_model()
        if callable(getattr(model, "get_base_model", None))
        else model
    )
    backbone = getattr(causal_model, "model", None)
    output_head = (
        causal_model.get_output_embeddings()
        if callable(getattr(causal_model, "get_output_embeddings", None))
        else None
    )
    if backbone is None or output_head is None:
        raise TypeError(
            "chunked exact loss requires a causal LM with a transformer body "
            "and output embeddings"
        )
    weight = getattr(output_head, "weight", None)
    if weight is None or getattr(weight, "ndim", None) != 2:
        raise TypeError("causal LM output embeddings must expose a 2D weight")
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = getattr(outputs, "last_hidden_state", None)
    if hidden_states is None:
        raise TypeError("causal LM transformer body returned no last hidden state")
    if hidden_states.shape[:-1] != input_ids.shape:
        raise ValueError("causal LM hidden-state dimensions do not match input IDs")
    if hidden_states.shape[-1] != weight.shape[-1]:
        raise ValueError("hidden size does not match the vocabulary projection")
    return CausalLMProjection(
        hidden_states=hidden_states,
        weight=weight,
        bias=getattr(output_head, "bias", None),
    )
