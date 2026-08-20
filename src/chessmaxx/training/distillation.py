"""Multi-PV policy targets derived from Stockfish centipawn utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from chessmaxx.evaluation.schema import TeacherMove
from chessmaxx.training.schema import TrainingExample
from chessmaxx.training.tokenize import (
    EncodedExample,
    TokenizerLike,
    encode_prompt_target,
)


@dataclass(frozen=True, slots=True)
class PolicyTarget:
    move: str
    score_cp: int
    probability: float


@dataclass(frozen=True, slots=True)
class EncodedPolicyExample:
    example_id: str
    candidates: tuple[EncodedExample, ...]
    teacher_probabilities: tuple[float, ...]

    @property
    def candidate_tokens(self) -> int:
        return sum(len(candidate.input_ids) for candidate in self.candidates)


@dataclass(frozen=True, slots=True)
class PolicyDatasetSummary:
    examples: int
    candidate_sequences: int
    mean_candidates_per_example: float
    mean_teacher_top1_probability: float
    mean_teacher_entropy: float
    maximum_candidate_length: int


def centipawn_policy(
    teacher_moves: Sequence[TeacherMove],
    *,
    temperature_cp: float,
    max_candidates: int,
) -> tuple[PolicyTarget, ...]:
    """Turn ranked engine utilities into a normalized candidate policy."""

    if temperature_cp <= 0:
        raise ValueError("temperature_cp must be positive")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    candidates = list(teacher_moves[:max_candidates])
    if not candidates:
        raise ValueError("teacher policy requires at least one move")
    maximum = max(move.score_cp for move in candidates)
    weights = [
        math.exp((move.score_cp - maximum) / temperature_cp)
        for move in candidates
    ]
    total = sum(weights)
    return tuple(
        PolicyTarget(
            move=move.move,
            score_cp=move.score_cp,
            probability=weight / total,
        )
        for move, weight in zip(candidates, weights, strict=True)
    )


def policy_entropy(policy: Sequence[PolicyTarget]) -> float:
    if not policy:
        raise ValueError("cannot measure an empty policy")
    return -sum(
        target.probability * math.log(target.probability)
        for target in policy
        if target.probability > 0
    )


def encode_policy_example(
    example: TrainingExample,
    tokenizer: TokenizerLike,
    *,
    max_length: int,
    temperature_cp: float,
    max_candidates: int,
) -> EncodedPolicyExample:
    policy = centipawn_policy(
        example.teacher_moves,
        temperature_cp=temperature_cp,
        max_candidates=max_candidates,
    )
    candidates = tuple(
        encode_prompt_target(
            example_id=f"{example.example_id}:{target.move}",
            fen=example.fen,
            move_uci=target.move,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        for target in policy
    )
    return EncodedPolicyExample(
        example_id=example.example_id,
        candidates=candidates,
        teacher_probabilities=tuple(target.probability for target in policy),
    )


class PolicyTokenDataset:
    def __init__(
        self,
        examples: list[TrainingExample],
        tokenizer: TokenizerLike,
        *,
        max_length: int,
        temperature_cp: float,
        max_candidates: int,
    ) -> None:
        self.items = [
            encode_policy_example(
                example,
                tokenizer,
                max_length=max_length,
                temperature_cp=temperature_cp,
                max_candidates=max_candidates,
            )
            for example in examples
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {
            "example_id": item.example_id,
            "input_ids": [list(candidate.input_ids) for candidate in item.candidates],
            "attention_mask": [
                list(candidate.attention_mask) for candidate in item.candidates
            ],
            "labels": [list(candidate.labels) for candidate in item.candidates],
            "teacher_probabilities": list(item.teacher_probabilities),
        }


def summarize_policy_dataset(dataset: PolicyTokenDataset) -> PolicyDatasetSummary:
    if not dataset.items:
        raise ValueError("cannot summarize an empty policy dataset")
    candidate_count = sum(len(item.candidates) for item in dataset.items)
    entropies = [
        -sum(
            probability * math.log(probability)
            for probability in item.teacher_probabilities
            if probability > 0
        )
        for item in dataset.items
    ]
    return PolicyDatasetSummary(
        examples=len(dataset.items),
        candidate_sequences=candidate_count,
        mean_candidates_per_example=candidate_count / len(dataset.items),
        mean_teacher_top1_probability=sum(
            item.teacher_probabilities[0] for item in dataset.items
        )
        / len(dataset.items),
        mean_teacher_entropy=sum(entropies) / len(entropies),
        maximum_candidate_length=max(
            len(candidate.input_ids)
            for item in dataset.items
            for candidate in item.candidates
        ),
    )


def pad_policy_records(
    records: list[dict[str, Any]], *, pad_token_id: int
) -> dict[str, Any]:
    """Pad candidate and token dimensions while preserving position groups."""

    if not records:
        raise ValueError("cannot collate an empty policy batch")
    max_candidates = max(len(record["input_ids"]) for record in records)
    max_length = max(
        len(candidate)
        for record in records
        for candidate in record["input_ids"]
    )
    batch: dict[str, Any] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "teacher_probabilities": [],
        "candidate_mask": [],
    }
    for record in records:
        candidate_count = len(record["input_ids"])
        if not candidate_count:
            raise ValueError("policy example contains no candidates")
        if not (
            len(record["attention_mask"])
            == len(record["labels"])
            == len(record["teacher_probabilities"])
            == candidate_count
        ):
            raise ValueError("policy candidate fields have different lengths")
        input_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for input_ids, attention_mask, labels in zip(
            record["input_ids"],
            record["attention_mask"],
            record["labels"],
            strict=True,
        ):
            padding = max_length - len(input_ids)
            input_rows.append(input_ids + [pad_token_id] * padding)
            attention_rows.append(attention_mask + [0] * padding)
            label_rows.append(labels + [-100] * padding)
        for _ in range(max_candidates - candidate_count):
            input_rows.append([pad_token_id] * max_length)
            attention_rows.append([1] + [0] * (max_length - 1))
            label_rows.append([-100] * max_length)
        batch["input_ids"].append(input_rows)
        batch["attention_mask"].append(attention_rows)
        batch["labels"].append(label_rows)
        batch["teacher_probabilities"].append(
            record["teacher_probabilities"] + [0.0] * (max_candidates - candidate_count)
        )
        batch["candidate_mask"].append(
            [True] * candidate_count + [False] * (max_candidates - candidate_count)
        )
    return batch


class PolicyCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("training requires the optional model dependencies") from exc
        padded = pad_policy_records(records, pad_token_id=self.pad_token_id)
        return {
            "input_ids": torch.tensor(padded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                padded["attention_mask"], dtype=torch.long
            ),
            "labels": torch.tensor(padded["labels"], dtype=torch.long),
            "teacher_probabilities": torch.tensor(
                padded["teacher_probabilities"], dtype=torch.float32
            ),
            "candidate_mask": torch.tensor(
                padded["candidate_mask"], dtype=torch.bool
            ),
        }


def candidate_sequence_log_likelihoods(logits: Any, labels: Any) -> tuple[Any, Any]:
    """Sum causal token log-probabilities for every candidate sequence."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("distillation loss requires PyTorch") from exc
    if logits.ndim != 4 or labels.ndim != 3:
        raise ValueError("expected logits [B,K,L,V] and labels [B,K,L]")
    if logits.shape[:3] != labels.shape:
        raise ValueError("logit and label dimensions do not match")
    shifted_logits = logits[..., :-1, :].float()
    shifted_labels = labels[..., 1:]
    supervised = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~supervised, 0)
    token_log_probabilities = functional.log_softmax(
        shifted_logits, dim=-1
    ).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    sequence_scores = (token_log_probabilities * supervised).sum(dim=-1)
    token_counts = supervised.sum(dim=-1)
    return sequence_scores, token_counts


def dense_policy_objective(
    sequence_log_likelihoods: Any,
    supervised_token_counts: Any,
    teacher_probabilities: Any,
    candidate_mask: Any,
    *,
    hard_loss_weight: float,
    student_temperature: float,
) -> dict[str, Any]:
    """Blend top-1 response loss with KL over candidate sequence scores."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("distillation loss requires PyTorch") from exc
    if not 0 <= hard_loss_weight <= 1:
        raise ValueError("hard_loss_weight must be in [0, 1]")
    if student_temperature <= 0:
        raise ValueError("student_temperature must be positive")
    expected_shape = sequence_log_likelihoods.shape
    if any(
        value.shape != expected_shape
        for value in (
            supervised_token_counts,
            teacher_probabilities,
            candidate_mask,
        )
    ):
        raise ValueError("policy tensors must share shape [B,K]")
    if torch.any(candidate_mask.sum(dim=-1) == 0):
        raise ValueError("every policy group must contain a candidate")
    if torch.any(supervised_token_counts.masked_select(candidate_mask) == 0):
        raise ValueError("every real candidate must contain a supervised token")

    masked_scores = (sequence_log_likelihoods / student_temperature).masked_fill(
        ~candidate_mask, float("-inf")
    )
    student_log_policy = functional.log_softmax(masked_scores, dim=-1)
    positive_teacher = teacher_probabilities > 0
    teacher_log_policy = torch.where(
        positive_teacher,
        teacher_probabilities.log(),
        torch.zeros_like(teacher_probabilities),
    )
    policy_loss = (
        teacher_probabilities
        * (teacher_log_policy - student_log_policy.masked_fill(~candidate_mask, 0.0))
    ).sum(dim=-1).mean()

    top_indices = teacher_probabilities.argmax(dim=-1, keepdim=True)
    top_scores = sequence_log_likelihoods.gather(-1, top_indices).squeeze(-1)
    top_token_counts = supervised_token_counts.gather(
        -1, top_indices
    ).squeeze(-1)
    hard_loss = (-top_scores / top_token_counts).mean()
    loss = hard_loss_weight * hard_loss + (1 - hard_loss_weight) * policy_loss
    return {
        "loss": loss,
        "hard_loss": hard_loss.detach(),
        "policy_loss": policy_loss.detach(),
        "student_log_policy": student_log_policy.detach(),
    }
