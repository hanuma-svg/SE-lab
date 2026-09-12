from __future__ import annotations

from typing import Any

from se_lab.experiments.contracts import ExperimentResult


def compare_experiments(left: ExperimentResult, right: ExperimentResult) -> dict[str, Any]:
    mismatches: list[str] = []
    if left.config_hash != right.config_hash:
        mismatches.append("Experiment configuration hashes differ.")
    if left.experiment_id != right.experiment_id:
        mismatches.append("Experiment IDs differ.")
    left_variants = set(left.aggregation.get("variants", {}))
    right_variants = set(right.aggregation.get("variants", {}))
    if left_variants != right_variants:
        mismatches.append("Variant sets differ.")
    return {
        "report_type": "comparison",
        "experiment_id": left.experiment_id,
        "comparable": not mismatches and left.has_matched_budgets and right.has_matched_budgets,
        "mismatches": mismatches + left.mismatches + right.mismatches,
        "left": left.aggregation,
        "right": right.aggregation,
        "scientific_note": "Comparison is descriptive and does not establish superiority without a pre-registered analysis and sufficient matched measurements.",
    }


def failure_analysis(result: ExperimentResult) -> dict[str, Any]:
    failures = [run for run in result.runs if not run.pass_at_1]
    return {
        "report_type": "failure_analysis",
        "experiment_id": result.experiment_id,
        "failure_count": len(failures),
        "by_class": result.aggregation.get("variants", {}),
        "runs": [run.model_dump(mode="json") for run in failures],
    }
