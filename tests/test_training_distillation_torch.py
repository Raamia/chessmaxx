import math

import pytest

torch = pytest.importorskip("torch")

from chessmaxx.training.distillation import (  # noqa: E402
    candidate_sequence_log_likelihoods,
    dense_policy_objective,
)


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
