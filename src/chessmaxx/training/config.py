"""Typed configuration for the tiny supervised fine-tuning proof."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TinySFTProfile:
    name: str
    model_id: str
    revision: str = "main"
    method: str = "lora"
    dtype: str = "bfloat16"
    max_examples: int = 100
    max_length: int = 256
    packing: bool = True
    isolate_packed_attention: bool = False
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    epochs: float = 20.0
    learning_rate: float = 0.0002
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    logging_steps: int = 1
    save_steps: int = 25
    seed: int = 2026
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: str = "all-linear"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.model_id.strip():
            raise ValueError("name and model_id must not be empty")
        if self.method not in {"lora", "full"}:
            raise ValueError("method must be lora or full")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if self.isolate_packed_attention and not self.packing:
            raise ValueError("isolated packed attention requires packing")
        for field_name in (
            "max_examples",
            "max_length",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "logging_steps",
            "save_steps",
            "lora_rank",
            "lora_alpha",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in ("epochs", "learning_rate", "max_grad_norm"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in ("weight_decay", "warmup_ratio", "lora_dropout"):
            if not 0 <= getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be in [0, 1)")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TinySFTProfile":
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown tiny-SFT setting(s): {', '.join(sorted(unknown))}")
        return cls(**value)


def load_tiny_sft_profile(path: str | Path) -> TinySFTProfile:
    source = Path(path)
    try:
        value = tomllib.loads(source.read_text(encoding="utf-8"))
        section = value["tiny_sft"]
        if not isinstance(section, dict):
            raise TypeError("[tiny_sft] must be a TOML table")
        return TinySFTProfile.from_dict(section)
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid tiny-SFT profile {source}: {exc}") from exc
