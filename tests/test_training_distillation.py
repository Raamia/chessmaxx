import math

import pytest

from chessmaxx.evaluation.schema import TeacherMove
from chessmaxx.training.distillation import centipawn_policy, policy_entropy


MOVES = (
    TeacherMove("e2e4", 80),
    TeacherMove("d2d4", 40),
    TeacherMove("g1f3", 0),
)


def test_centipawn_policy_is_normalized_and_ranked():
    policy = centipawn_policy(MOVES, temperature_cp=100.0, max_candidates=3)

    assert [target.move for target in policy] == ["e2e4", "d2d4", "g1f3"]
    assert sum(target.probability for target in policy) == pytest.approx(1.0)
    assert policy[0].probability > policy[1].probability > policy[2].probability


def test_temperature_controls_teacher_sharpness():
    cold = centipawn_policy(MOVES, temperature_cp=25.0, max_candidates=3)
    warm = centipawn_policy(MOVES, temperature_cp=200.0, max_candidates=3)

    assert cold[0].probability > warm[0].probability
    assert policy_entropy(cold) < policy_entropy(warm)


def test_truncation_renormalizes_the_retained_candidates():
    policy = centipawn_policy(MOVES, temperature_cp=100.0, max_candidates=2)

    assert len(policy) == 2
    assert sum(target.probability for target in policy) == pytest.approx(1.0)


def test_equal_scores_produce_a_uniform_policy():
    policy = centipawn_policy(
        (TeacherMove("e2e4", 10), TeacherMove("d2d4", 10)),
        temperature_cp=50.0,
        max_candidates=2,
    )

    assert [target.probability for target in policy] == pytest.approx([0.5, 0.5])
    assert policy_entropy(policy) == pytest.approx(math.log(2))


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_policy_rejects_nonpositive_temperature(temperature):
    with pytest.raises(ValueError, match="temperature_cp"):
        centipawn_policy(MOVES, temperature_cp=temperature, max_candidates=3)
