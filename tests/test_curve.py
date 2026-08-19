import pytest

from chessmaxx.evaluation.curve import build_checkpoint_curve


def evaluation_report(*, mode, legal, top1, label=None, step=None, dataset="data"):
    return {
        "settings": {
            "mode": mode,
            "dataset_sha256": dataset,
            "checkpoint_label": label,
            "checkpoint_step": step,
        },
        "summary": {
            "parse_rate": legal,
            "legal_move_rate": legal,
            "top1_agreement_rate": top1,
        },
    }


def training_report():
    return {
        "dataset_sha256": "data",
        "profile": {"name": "scaled"},
        "runtime": {"optimizer_steps": 100, "wall_seconds": 200.0},
        "trainer_metrics": {"total_flos": 1_000.0},
        "learning_curve": [
            {"step": 25, "loss": 1.0, "eval_loss": 1.1},
            {"step": 50, "loss": 0.7, "eval_loss": 0.9},
            {"step": 100, "loss": 0.4, "eval_loss": 0.8},
        ],
    }


def test_builds_base_to_final_capability_curve():
    curve = build_checkpoint_curve(
        evaluation_report(mode="labelled_base", legal=0.0, top1=0.0),
        training_report(),
        [
            evaluation_report(
                mode="adapter", legal=0.4, top1=0.2, label="checkpoint-50", step=50
            ),
            evaluation_report(
                mode="adapter", legal=0.8, top1=0.5, label="final", step=None
            ),
        ],
    )

    assert [point["step"] for point in curve["points"]] == [0, 50, 100]
    assert curve["points"][1]["estimated_cumulative_training_seconds"] == 100.0
    assert curve["points"][1]["estimated_cumulative_training_flops"] == 500.0
    assert curve["points"][1]["eval_loss"] == 0.9
    assert curve["points"][2]["legal_move_rate"] == 0.8


def test_curve_rejects_different_datasets():
    with pytest.raises(ValueError, match="do not use the training dataset"):
        build_checkpoint_curve(
            evaluation_report(
                mode="labelled_base", legal=0.0, top1=0.0, dataset="other"
            ),
            training_report(),
            [],
        )
