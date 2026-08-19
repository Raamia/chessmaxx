"""Typed configuration for repeatable model baselines."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_DTYPES = {"auto", "float16", "bfloat16", "float32"}


@dataclass(frozen=True, slots=True)
class BaselineProfile:
    """Model and runtime settings that define one baseline evaluation."""

    name: str
    model_id: str
    revision: str = "main"
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 8
    max_new_tokens: int = 8
    stockfish_nodes: int = 50_000
    stockfish_multipv: int = 3
    stockfish_threads: int = 1
    stockfish_hash_mb: int = 64

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("profile name must not be empty")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not self.revision.strip():
            raise ValueError("revision must not be empty")
        if self.dtype not in SUPPORTED_DTYPES:
            choices = ", ".join(sorted(SUPPORTED_DTYPES))
            raise ValueError(f"dtype must be one of: {choices}")
        for field_name in (
            "batch_size",
            "max_new_tokens",
            "stockfish_nodes",
            "stockfish_multipv",
            "stockfish_threads",
            "stockfish_hash_mb",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BaselineProfile":
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown baseline setting(s): {names}")
        return cls(**value)


def load_baseline_profile(path: str | Path) -> BaselineProfile:
    source = Path(path)
    try:
        value = tomllib.loads(source.read_text(encoding="utf-8"))
        profile = value["baseline"]
        if not isinstance(profile, dict):
            raise TypeError("[baseline] must be a TOML table")
        return BaselineProfile.from_dict(profile)
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid baseline profile {source}: {exc}") from exc
