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
    objective: str = "hard_sft"
    dtype: str = "bfloat16"
    max_examples: int = 100
    max_validation_examples: int = 0
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
    evaluation_steps: int = 25
    save_steps: int = 25
    save_total_limit: int = 2
    seed: int = 2026
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: str = "all-linear"
    teacher_temperature_cp: float = 100.0
    max_teacher_candidates: int = 3
    hard_loss_weight: float = 0.5
    student_temperature: float = 1.0
    policy_loss_backend: str = "dense"
    vocabulary_chunk_size: int = 4096

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.model_id.strip():
            raise ValueError("name and model_id must not be empty")
        if self.method not in {"lora", "full"}:
            raise ValueError("method must be lora or full")
        if self.objective not in {"hard_sft", "multipv_policy"}:
            raise ValueError("objective must be hard_sft or multipv_policy")
        if self.policy_loss_backend not in {"dense", "chunked_exact"}:
            raise ValueError(
                "policy_loss_backend must be dense or chunked_exact"
            )
        if self.objective != "multipv_policy" and self.policy_loss_backend != "dense":
            raise ValueError(
                "chunked_exact policy loss requires the multi-PV policy objective"
            )
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if self.isolate_packed_attention and not self.packing:
            raise ValueError("isolated packed attention requires packing")
        if self.objective == "multipv_policy" and self.packing:
            raise ValueError("multi-PV policy training does not support packing")
        for field_name in (
            "max_examples",
            "max_length",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "logging_steps",
            "evaluation_steps",
            "save_steps",
            "save_total_limit",
            "lora_rank",
            "lora_alpha",
            "max_teacher_candidates",
            "vocabulary_chunk_size",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.max_validation_examples < 0:
            raise ValueError("max_validation_examples must be non-negative")
        for field_name in (
            "epochs",
            "learning_rate",
            "max_grad_norm",
            "teacher_temperature_cp",
            "student_temperature",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in ("weight_decay", "warmup_ratio", "lora_dropout"):
            if not 0 <= getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be in [0, 1)")
        if not 0 <= self.hard_loss_weight <= 1:
            raise ValueError("hard_loss_weight must be in [0, 1]")

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
