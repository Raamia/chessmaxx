import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from chessmaxx.training.distillation import (  # noqa: E402
    candidate_sequence_log_likelihoods,
    dense_policy_objective,
)
from chessmaxx.training.config import TinySFTProfile  # noqa: E402
from chessmaxx.training.train import _policy_trainer_class  # noqa: E402


def test_candidate_scores_sum_only_supervised_next_tokens():
    logits = torch.zeros((1, 2, 4, 5))
    labels = torch.tensor(
        [[[-100, -100, 2, 3], [-100, -100, -100, 4]]]
    )

    scores, counts = candidate_sequence_log_likelihoods(logits, labels)

    assert counts.tolist() == [[2, 1]]
    assert scores.tolist()[0] == pytest.approx(
        [-2 * math.log(5), -math.log(5)]
    )


def test_policy_kl_is_zero_when_student_matches_teacher():
    teacher = torch.tensor([[0.7, 0.3]])
    result = dense_policy_objective(
        teacher.log(),
        torch.ones((1, 2)),
        teacher,
        torch.tensor([[True, True]]),
        hard_loss_weight=0.0,
        student_temperature=1.0,
    )

    assert result["policy_loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert result["loss"].item() == pytest.approx(0.0, abs=1e-6)


def test_policy_objective_ignores_padded_candidates():
    result = dense_policy_objective(
        torch.tensor([[-0.2, -0.5, -100.0]]),
        torch.tensor([[2, 2, 0]]),
        torch.tensor([[0.6, 0.4, 0.0]]),
        torch.tensor([[True, True, False]]),
        hard_loss_weight=0.5,
        student_temperature=1.0,
    )

    assert torch.isfinite(result["loss"])
    assert result["student_log_policy"][0, 2].item() == float("-inf")


def test_policy_trainer_backpropagates_through_candidate_logits():
    class BaseTrainer:
        pass

    logits = torch.randn((2, 4, 7), requires_grad=True)

    class Model:
        def __call__(self, **kwargs):
            assert kwargs["input_ids"].shape == (2, 4)
            return SimpleNamespace(logits=logits)

    profile = TinySFTProfile(
        name="policy",
        model_id="model",
        objective="multipv_policy",
        packing=False,
    )
    trainer_type = _policy_trainer_class(
        SimpleNamespace(Trainer=BaseTrainer), profile
    )
    trainer = trainer_type()
    loss = trainer.compute_loss(
        Model(),
        {
            "input_ids": torch.ones((1, 2, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 2, 4), dtype=torch.long),
            "labels": torch.tensor(
                [[[-100, -100, 2, 3], [-100, -100, 4, 5]]]
            ),
            "teacher_probabilities": torch.tensor([[0.6, 0.4]]),
            "candidate_mask": torch.tensor([[True, True]]),
        },
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
