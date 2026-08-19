"""Checkpointed tiny-SFT execution with 8 GB GPU telemetry."""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from chessmaxx.training.config import TinySFTProfile
from chessmaxx.training.dataset import load_training_examples
from chessmaxx.training.packing import (
    CausalLMCollator,
    PackedTokenDataset,
    pack_encoded_examples,
)
from chessmaxx.training.schema import TrainingExample
from chessmaxx.training.tokenize import SupervisedTokenDataset


@dataclass(frozen=True, slots=True)
class TrainingDataSummary:
    available_train_examples: int
    selected_examples: int
    optimizer_records: int
    input_tokens_per_epoch: int
    supervised_tokens_per_epoch: int


def select_training_examples(
    examples: list[TrainingExample], *, maximum: int
) -> list[TrainingExample]:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    selected = [example for example in examples if example.split == "train"][:maximum]
    if not selected:
        raise ValueError("dataset contains no training examples")
    return selected


def prepare_training_dataset(
    examples: list[TrainingExample],
    tokenizer: Any,
    *,
    max_length: int,
    packing: bool,
) -> tuple[Any, TrainingDataSummary]:
    selected = SupervisedTokenDataset(examples, tokenizer, max_length=max_length)
    input_tokens = sum(len(item.input_ids) for item in selected.items)
    supervised_tokens = sum(item.supervised_tokens for item in selected.items)
    if packing:
        records = PackedTokenDataset(
            pack_encoded_examples(selected.items, max_length=max_length)
        )
    else:
        records = selected
    return records, TrainingDataSummary(
        available_train_examples=len(examples),
        selected_examples=len(examples),
        optimizer_records=len(records),
        input_tokens_per_epoch=input_tokens,
        supervised_tokens_per_epoch=supervised_tokens,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dtype(torch: Any, name: str) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _warmup_arguments(
    training_arguments: Any, warmup_ratio: float
) -> dict[str, float]:
    parameters = inspect.signature(training_arguments).parameters
    if "warmup_ratio" in parameters:
        return {"warmup_ratio": warmup_ratio}
    if "warmup_steps" in parameters:
        return {"warmup_steps": warmup_ratio}
    raise RuntimeError(
        "TrainingArguments accepts neither warmup_ratio nor warmup_steps"
    )


def run_tiny_sft(
    profile: TinySFTProfile,
    *,
    profile_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train on CUDA and return the report also written below ``output_dir``."""

    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "training requires `pip install -e '.[train]'`"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("tiny SFT requires CUDA; refusing to fall back to CPU")

    profile_source = Path(profile_path).resolve()
    dataset_source = Path(dataset_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    all_examples = load_training_examples(dataset_source)
    available_train = sum(example.split == "train" for example in all_examples)
    examples = select_training_examples(all_examples, maximum=profile.max_examples)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        profile.model_id,
        revision=profile.revision,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("model tokenizer does not define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        profile.model_id,
        revision=profile.revision,
        dtype=_dtype(torch, profile.dtype),
    )
    resolved_model_revision = getattr(model.config, "_commit_hash", None)
    model.config.use_cache = False
    if profile.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if profile.method == "lora":
        lora_config = peft.LoraConfig(
            r=profile.lora_rank,
            lora_alpha=profile.lora_alpha,
            lora_dropout=profile.lora_dropout,
            target_modules=profile.lora_target_modules,
            bias="none",
            task_type=peft.TaskType.CAUSAL_LM,
        )
        model = peft.get_peft_model(model, lora_config)
        if profile.gradient_checkpointing:
            model.enable_input_require_grads()

    dataset, data_summary = prepare_training_dataset(
        examples,
        tokenizer,
        max_length=profile.max_length,
        packing=profile.packing,
    )
    data_summary = TrainingDataSummary(
        available_train_examples=available_train,
        selected_examples=data_summary.selected_examples,
        optimizer_records=data_summary.optimizer_records,
        input_tokens_per_epoch=data_summary.input_tokens_per_epoch,
        supervised_tokens_per_epoch=data_summary.supervised_tokens_per_epoch,
    )

    use_bf16 = profile.dtype == "bfloat16"
    use_fp16 = profile.dtype == "float16"
    arguments = transformers.TrainingArguments(
        output_dir=str(destination / "checkpoints"),
        per_device_train_batch_size=profile.per_device_batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        num_train_epochs=profile.epochs,
        learning_rate=profile.learning_rate,
        weight_decay=profile.weight_decay,
        **_warmup_arguments(
            transformers.TrainingArguments, profile.warmup_ratio
        ),
        max_grad_norm=profile.max_grad_norm,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=profile.gradient_checkpointing,
        logging_strategy="steps",
        logging_steps=profile.logging_steps,
        save_strategy="steps",
        save_steps=profile.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=profile.seed,
        data_seed=profile.seed,
        optim="adamw_torch",
        include_num_input_tokens_seen=True,
    )
    trainer = transformers.Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = trainer.train(
        resume_from_checkpoint=(
            str(Path(resume_from_checkpoint).resolve())
            if resume_from_checkpoint is not None
            else None
        )
    )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started

    final_dir = destination / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    metrics = dict(result.metrics)
    tokens_seen = metrics.get("num_input_tokens_seen")
    if tokens_seen is None:
        tokens_seen = data_summary.input_tokens_per_epoch * profile.epochs
    report: dict[str, Any] = {
        "schema_version": 1,
        "profile": asdict(profile),
        "profile_path": str(profile_source),
        "profile_sha256": _sha256(profile_source),
        "dataset_path": str(dataset_source),
        "dataset_sha256": _sha256(dataset_source),
        "git_commit": _git_commit(),
        "resolved_model_revision": resolved_model_revision,
        "data": asdict(data_summary),
        "parameters": {
            "total": parameter_count,
            "trainable": trainable_parameter_count,
            "trainable_fraction": trainable_parameter_count / parameter_count,
        },
        "runtime": {
            "wall_seconds": wall_seconds,
            "input_tokens_seen": tokens_seen,
            "input_tokens_per_second": tokens_seen / wall_seconds,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "trainer_metrics": metrics,
        "final_model_dir": str(final_dir),
    }
    (destination / "training-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
