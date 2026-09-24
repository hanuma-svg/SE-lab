from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from statistics import mean, median

from se_lab.experiments.contracts import RunRecord


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a deterministic two-sided Wilson interval for a binomial rate."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) / trials) + (z * z / (4.0 * trials * trials))) / denominator
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def summarize_runs(runs: Iterable[RunRecord]) -> dict[str, object]:
    materialized = list(runs)
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for run in materialized:
        grouped[run.variant].append(run)
    variants: dict[str, object] = {}
    for variant, items in sorted(grouped.items()):
        successes = sum(item.pass_at_1 for item in items)
        durations = [item.wall_clock_duration_seconds for item in items if item.wall_clock_duration_seconds is not None]
        low, high = wilson_interval(successes, len(items))
        variants[variant] = {
            "sample_count": len(items),
            "successes": successes,
            "pass_rate": successes / len(items) if items else 0.0,
            "pass_rate_ci95": {"low": low, "high": high},
            "failure_classes": _failure_classes(items),
            "mean_model_calls": _mean([item.model_calls for item in items]),
            "mean_tool_calls": _mean([item.tool_calls for item in items]),
            "mean_retries": _mean([item.retries for item in items]),
            "mean_wall_clock_duration_seconds": _mean(durations),
            "median_wall_clock_duration_seconds": median(durations) if durations else None,
        }
    return {
        "sample_count": sum(len(items) for items in grouped.values()),
        "variants": variants,
        "paired_pass_differences": paired_pass_differences(materialized),
        "statistical_note": "Wilson intervals are descriptive and do not establish superiority or causal significance.",
    }


def paired_pass_differences(runs: list[RunRecord]) -> list[dict[str, object]]:
    by_key: dict[tuple[int, str], dict[str, RunRecord]] = defaultdict(dict)
    for run in runs:
        by_key[(run.repetition, run.ablation)][run.variant] = run
    differences: list[dict[str, object]] = []
    for (repetition, ablation), variants in sorted(by_key.items()):
        baseline = variants.get("baseline")
        treatment = variants.get("multi_agent")
        if baseline is None or treatment is None:
            continue
        differences.append({
            "repetition": repetition,
            "ablation": ablation,
            "baseline_pass_at_1": baseline.pass_at_1,
            "multi_agent_pass_at_1": treatment.pass_at_1,
            "difference": int(treatment.pass_at_1) - int(baseline.pass_at_1),
            "budget_match": baseline.budget_match and treatment.budget_match,
        })
    return differences


def _failure_classes(items: list[RunRecord]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item.failure_class] += 1
    return dict(sorted(counts.items()))


def _mean(values: list[float | int]) -> float | None:
    return mean(values) if values else None
