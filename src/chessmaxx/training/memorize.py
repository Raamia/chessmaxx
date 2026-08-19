"""Diagnostic evaluation on the exact examples used by tiny SFT."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chessmaxx.evaluation.model import HuggingFaceMoveGenerator, MoveGenerator
from chessmaxx.evaluation.moves import check_generated_move
from chessmaxx.training.config import TinySFTProfile
from chessmaxx.training.dataset import load_training_examples
from chessmaxx.training.schema import TrainingExample
from chessmaxx.training.train import select_training_examples


@dataclass(frozen=True, slots=True)
class MemorizationResult:
    example_id: str
    fen: str
    target_move: str
    raw_output: str
    parsed_move: str | None
    is_legal: bool
    target_move_match: bool
    exact_response_match: bool
    error: str | None


def summarize_memorization(
    results: list[MemorizationResult],
) -> dict[str, int | float]:
    if not results:
        raise ValueError("cannot summarize zero memorization results")
    count = len(results)
    parsed = sum(result.parsed_move is not None for result in results)
    legal = sum(result.is_legal for result in results)
    target = sum(result.target_move_match for result in results)
    exact = sum(result.exact_response_match for result in results)
    return {
        "examples": count,
        "parsed_moves": parsed,
        "parse_rate": parsed / count,
        "legal_moves": legal,
        "legal_move_rate": legal / count,
        "target_move_matches": target,
        "target_move_accuracy": target / count,
        "exact_response_matches": exact,
        "exact_response_accuracy": exact / count,
    }


def evaluate_memorization(
    examples: list[TrainingExample],
    generator: MoveGenerator,
    *,
    batch_size: int,
) -> tuple[list[MemorizationResult], dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    generator.reset_telemetry()
    results: list[MemorizationResult] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        generated = generator.generate_many([example.fen for example in batch])
        if len(generated) != len(batch):
            raise RuntimeError("model returned a different number of outputs than inputs")
        for example, response in zip(batch, generated, strict=True):
            checked = check_generated_move(example.fen, response.raw_output)
            results.append(
                MemorizationResult(
                    example_id=example.example_id,
                    fen=example.fen,
                    target_move=example.target_move,
                    raw_output=response.raw_output,
                    parsed_move=checked.parsed_move,
                    is_legal=checked.is_legal,
                    target_move_match=checked.parsed_move == example.target_move,
                    exact_response_match=(
                        response.raw_output.strip().lower() == example.target_move
                    ),
                    error=checked.error,
                )
            )
    return results, dict(generator.telemetry)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_lora_generator(
    profile: TinySFTProfile,
    *,
    adapter_dir: str | Path,
    device: str = "cuda",
    max_new_tokens: int = 8,
) -> HuggingFaceMoveGenerator:
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "adapter evaluation requires `pip install -e '.[train]'`"
        ) from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    selected_dtype = getattr(torch, profile.dtype)
    adapter = Path(adapter_dir).resolve()
    tokenizer = transformers.AutoTokenizer.from_pretrained(adapter)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        profile.model_id,
        revision=profile.revision,
        dtype=selected_dtype,
    )
    model = peft.PeftModel.from_pretrained(base_model, adapter).to(device)
    return HuggingFaceMoveGenerator(
        model=model,
        tokenizer=tokenizer,
        model_name=str(adapter),
        device=device,
        max_new_tokens=max_new_tokens,
        revision=profile.revision,
        transformers_version=transformers.__version__,
    )


def run_memorization_check(
    profile: TinySFTProfile,
    *,
    profile_path: str | Path,
    dataset_path: str | Path,
    adapter_dir: str | Path,
    output_path: str | Path,
    batch_size: int,
    device: str = "cuda",
) -> dict[str, Any]:
    profile_source = Path(profile_path).resolve()
    dataset_source = Path(dataset_path).resolve()
    adapter = Path(adapter_dir).resolve()
    examples = select_training_examples(
        load_training_examples(dataset_source), maximum=profile.max_examples
    )
    generator = load_lora_generator(profile, adapter_dir=adapter, device=device)
    results, telemetry = evaluate_memorization(
        examples, generator, batch_size=batch_size
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_kind": "training_set_memorization",
        "generalization_metric": False,
        "profile_path": str(profile_source),
        "profile_sha256": _file_sha256(profile_source),
        "dataset_path": str(dataset_source),
        "dataset_sha256": _file_sha256(dataset_source),
        "adapter_dir": str(adapter),
        "model": generator.metadata,
        "settings": {"batch_size": batch_size, "device": device},
        "telemetry": telemetry,
        "summary": summarize_memorization(results),
        "results": [asdict(result) for result in results],
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
