"""Exact policy loss primitives that avoid dense sequence logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CausalLMProjection:
    hidden_states: Any
    weight: Any
    bias: Any | None


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
