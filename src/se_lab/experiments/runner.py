from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from se_lab.agents import (
    BaselineConfig,
    MultiAgentConfig,
    MultiAgentWorkflow,
    SingleAgentBaseline,
)
from se_lab.evaluation import EvaluationTask, evaluate_events
from se_lab.experiments.contracts import ExperimentConfig, ExperimentResult, RunRecord
from se_lab.experiments.statistics import summarize_runs
from se_lab.models.provider import BudgetedModelProvider, create_provider


class ExperimentRunner:
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        task = EvaluationTask.from_file(config.task_path)
        config_hash = self._hash(
            {
                "config": config.model_dump(mode="json"),
                "task": task.to_dict(),
            }
        )
        self._ensure_fresh_evidence(config)
        runs: list[RunRecord] = []
        mismatches: list[str] = []
        for ablation in config.ablations:
            if not ablation.enabled:
                continue
            if ablation.name in {"no_reviewer", "no_handoffs"}:
                mismatches.append(f"Ablation {ablation.name} is not supported by the bounded Phase 4 graph.")
                continue
            for repetition in range(1, config.repetitions + 1):
                seed = config.seed + repetition - 1
                for variant in config.variants:
                    runs.append(self._run_one(config, task, config_hash, variant, ablation.name, repetition, seed))
        for run in runs:
            if run.model_calls > config.budget.max_model_calls:
                run.budget_match = False
                mismatches.append(f"{run.run_id} exceeded max_model_calls.")
            if run.tool_calls > config.budget.max_tool_calls:
                run.budget_match = False
                mismatches.append(f"{run.run_id} exceeded max_tool_calls.")
            if run.retries > config.budget.max_retries:
                run.budget_match = False
                mismatches.append(f"{run.run_id} exceeded max_retries.")
        aggregation = aggregate_runs(runs)
        return ExperimentResult(
            experiment_id=config.experiment_id,
            config_hash=config_hash,
            runs=runs,
            mismatches=mismatches,
            aggregation=aggregation,
            report=build_experiment_report(config, task, runs, mismatches, aggregation),
        )

    @staticmethod
    def _ensure_fresh_evidence(config: ExperimentConfig) -> None:
        for ablation in config.ablations:
            if not ablation.enabled or ablation.name in {"no_reviewer", "no_handoffs"}:
                continue
            for repetition in range(1, config.repetitions + 1):
                for variant in config.variants:
                    run_id = f"{config.experiment_id}-{variant}-{ablation.name}-r{repetition}"
                    evidence_root = Path(config.evidence_dir) / config.experiment_id / run_id
                    if evidence_root.exists():
                        raise FileExistsError(
                            f"Evidence already exists for run {run_id}: {evidence_root}. "
                            "Choose a new experiment_id or evidence_dir; historical evidence was preserved."
                        )

    def _run_one(
        self,
        config: ExperimentConfig,
        task: EvaluationTask,
        config_hash: str,
        variant: str,
        ablation: str,
        repetition: int,
        seed: int,
    ) -> RunRecord:
        run_id = f"{config.experiment_id}-{variant}-{ablation}-r{repetition}"
        evidence_root = Path(config.evidence_dir) / config.experiment_id / run_id
        events_dir = evidence_root / "events"
        artifacts_dir = evidence_root / "artifacts"
        provider = None
        if config.provider_name != "mock":
            provider = BudgetedModelProvider(
                create_provider(config.provider_name),
                max_calls=config.budget.max_model_calls,
                max_input_tokens=config.budget.max_input_tokens,
                max_output_tokens=config.budget.max_output_tokens,
                max_total_tokens=config.budget.max_total_tokens,
            )
        started = time.monotonic()
        if variant == "baseline":
            result = SingleAgentBaseline().run(
                task,
                config=BaselineConfig(
                    run_id=run_id,
                    seed=seed,
                    max_model_calls=config.budget.max_model_calls,
                    max_wall_clock=config.budget.max_wall_clock,
                    max_retries=0 if ablation == "no_retries" else config.budget.max_retries,
                    max_tokens=config.budget.max_tokens,
                    provider=provider,
                    provider_name=config.provider_name,
                    provider_version=config.provider_version,
                    model_name=config.model_name,
                    events_dir=events_dir,
                    artifacts_dir=artifacts_dir,
                ),
            )
            status = result.status
            summary = result.summary
            events = result.events
            model_calls = sum(1 for event in events if event.event_type == "ModelResponseReceived")
            tool_calls = sum(1 for event in events if event.event_type in {"ToolStarted", "ToolAuthorizationRequested"})
            retries = 0
        elif variant == "multi_agent":
            result = MultiAgentWorkflow().run(
                task,
                config=MultiAgentConfig(
                    run_id=run_id,
                    seed=seed,
                    max_model_calls=config.budget.max_model_calls,
                    max_tool_calls=config.budget.max_tool_calls,
                    max_wall_clock=config.budget.max_wall_clock,
                    max_retries=0 if ablation == "no_retries" else config.budget.max_retries,
                    provider=provider,
                    provider_name=config.provider_name,
                    provider_version=config.provider_version,
                    model_name=config.model_name,
                    events_dir=events_dir,
                    artifacts_dir=artifacts_dir,
                ),
            )
            status = result.status
            summary = result.summary
            events = result.events
            model_calls = sum(1 for event in events if event.event_type == "ModelResponseReceived")
            tool_calls = int(result.budgets.get("tool_calls_used", 0))
            retries = int(result.budgets.get("retries_used", 0))
        else:
            raise ValueError(f"Unsupported experiment variant: {variant}")
        audit = evaluate_events(run_id, events)
        duration = time.monotonic() - started
        failure_class = audit.primary_failure.value if audit.primary_failure else status
        return RunRecord(
            experiment_id=config.experiment_id,
            run_id=run_id,
            repetition=repetition,
            variant=variant,
            ablation=ablation,
            seed=seed,
            status=status,
            summary=summary,
            failure_class=failure_class,
            pass_at_1=status == "PASS",
            model_calls=model_calls,
            tool_calls=tool_calls,
            retries=retries,
            wall_clock_duration_seconds=round(duration, 6),
            budget_match=True,
            evidence_digest=audit.deterministic_digest,
            evidence_dir=str(evidence_root),
        )

    @staticmethod
    def _hash(value: Any) -> str:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def aggregate_runs(runs: list[RunRecord]) -> dict[str, Any]:
    return summarize_runs(runs)


def build_experiment_report(
    config: ExperimentConfig,
    task: EvaluationTask,
    runs: list[RunRecord],
    mismatches: list[str],
    aggregation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "report_type": "experiment",
        "experiment_id": config.experiment_id,
        "task_path": str(Path(config.task_path)),
        "task_id": task.task_id,
        "task_base_commit": task.commit_sha,
        "provider": {
            "name": config.provider_name,
            "version": config.provider_version,
            "model": config.model_name,
        },
        "seed": config.seed,
        "safety": {
            "allowed_write_paths": task.allowed_write_paths,
            "protected_paths": task.protected_paths,
            "network_enabled": False,
        },
        "variants": config.variants,
        "repetitions": config.repetitions,
        "matched_budget": config.budget.model_dump(mode="json"),
        "matched_budget_valid": not mismatches and all(run.budget_match for run in runs),
        "mismatches": mismatches,
        "runs": [run.model_dump(mode="json") for run in runs],
        "aggregation": aggregation,
        "scientific_note": "Descriptive measurements only; no claim of superiority is made without matched measured evidence.",
    }
