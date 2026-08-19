"""Join training telemetry and checkpoint evaluations into a learning curve."""

from __future__ import annotations

from typing import Any


CAPABILITY_KEYS = (
    "parse_rate",
    "legal_move_rate",
    "top1_agreement_rate",
    "topk_agreement_rate",
    "average_centipawn_regret",
    "blunder_100_rate",
    "blunder_300_rate",
    "blunder_500_rate",
)


def _dataset_hash(report: dict[str, Any]) -> str | None:
    return report.get("dataset_sha256") or report.get("settings", {}).get(
        "dataset_sha256"
    )


def _losses_at_step(
    learning_curve: list[dict[str, Any]], step: int
) -> dict[str, float]:
    eligible = [point for point in learning_curve if point.get("step", 0) <= step]
    if not eligible:
        return {}
    latest = max(eligible, key=lambda point: point["step"])
    return {
        key: float(latest[key])
        for key in ("loss", "eval_loss")
        if key in latest
    }


def build_checkpoint_curve(
    base_report: dict[str, Any],
    training_report: dict[str, Any],
    checkpoint_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_hash = _dataset_hash(training_report)
    if not dataset_hash:
        raise ValueError("training report has no dataset fingerprint")
    reports = [base_report, *checkpoint_reports]
    mismatched = [
        _dataset_hash(report)
        for report in reports
        if _dataset_hash(report) != dataset_hash
    ]
    if mismatched:
        raise ValueError("curve reports do not use the training dataset")

    total_steps = int(training_report["runtime"]["optimizer_steps"])
    if total_steps <= 0:
        raise ValueError("training report has no completed optimizer steps")
    wall_seconds = float(training_report["runtime"]["wall_seconds"])
    total_flops = float(training_report["trainer_metrics"].get("total_flos", 0.0))
    loss_curve = list(training_report.get("learning_curve", []))

    points: list[dict[str, Any]] = [
        {
            "label": "base",
            "step": 0,
            "estimated_cumulative_training_seconds": 0.0,
            "estimated_cumulative_training_flops": 0.0,
            **{
                key: base_report["summary"].get(key) for key in CAPABILITY_KEYS
            },
        }
    ]
    for report in checkpoint_reports:
        settings = report.get("settings", {})
        step_value = settings.get("checkpoint_step")
        label = settings.get("checkpoint_label") or "adapter"
        step = total_steps if step_value is None else int(step_value)
        if not 0 < step <= total_steps:
            raise ValueError(f"checkpoint {label!r} has invalid step {step}")
        fraction = step / total_steps
        points.append(
            {
                "label": label,
                "step": step,
                "estimated_cumulative_training_seconds": wall_seconds * fraction,
                "estimated_cumulative_training_flops": total_flops * fraction,
                **_losses_at_step(loss_curve, step),
                **{
                    key: report["summary"].get(key) for key in CAPABILITY_KEYS
                },
            }
        )
    points.sort(key=lambda point: (point["step"], point["label"] == "final"))
    return {
        "schema_version": 1,
        "dataset_sha256": dataset_hash,
        "training_profile": training_report["profile"],
        "training_wall_seconds": wall_seconds,
        "training_total_flops": total_flops,
        "time_axis_is_estimated_from_step_fraction": True,
        "points": points,
    }
