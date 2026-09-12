from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from se_lab.agents import (
    BaselineConfig,
    MultiAgentConfig,
    MultiAgentWorkflow,
    SingleAgentBaseline,
)
from se_lab.artifacts.store import ArtifactStore
from se_lab.benchmarks import BenchmarkCatalog
from se_lab.contracts import EventEnvelope, RunManifest
from se_lab.evaluation import EvaluationTask, IndependentEvaluator, evaluate_run
from se_lab.events.store import EventStore
from se_lab.experiments import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    compare_experiments,
    failure_analysis,
)
from se_lab.replay.replay import record_run, replay_run
from se_lab.replay.store import RecordStore
from se_lab.reporting.report import build_report


def _validate_manifest(path: str) -> int:
    try:
        manifest = RunManifest.read_from_file(path)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}))
        return 1

    print(
        json.dumps(
            {
                "valid": True,
                "run_id": manifest.run_id,
                "manifest_hash": manifest.content_hash(),
            }
        )
    )
    return 0


def _run_command(manifest_path: str, events_dir: str, artifacts_dir: str) -> int:
    manifest = RunManifest.read_from_file(manifest_path)
    event_store = EventStore(events_dir)
    artifact_store = ArtifactStore(artifacts_dir)

    event_store.append(
        EventEnvelope(
            run_id=manifest.run_id,
            event_type="RunCreated",
            role="system",
            payload={"task_id": manifest.task_id},
        )
    )

    artifact_store.store_bytes(manifest.model_dump_json().encode("utf-8"), object_type="manifest")
    report = build_report(manifest.run_id, event_store, artifact_store)

    report_dir = Path(events_dir).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"{manifest.run_id}.report.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _report_command(
    run_id: str | None,
    events_dir: str = ".se-lab/events",
    artifacts_dir: str = ".se-lab/artifacts",
    experiment_result: str | None = None,
    report_kind: str = "experiment",
) -> int:
    if experiment_result:
        result = ExperimentResult.model_validate_json(Path(experiment_result).read_text(encoding="utf-8"))
        payload = result.report if report_kind == "experiment" else failure_analysis(result)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if not run_id:
        raise ValueError("Either --run-id or --experiment-result is required")
    event_store = EventStore(events_dir)
    artifact_store = ArtifactStore(artifacts_dir)
    report_path = Path(events_dir).parent / f"{run_id}.report.json"

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = build_report(run_id, event_store, artifact_store)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _record_command(
    run_id: str,
    events_dir: str = ".se-lab/events",
    artifacts_dir: str = ".se-lab/artifacts",
    records_dir: str = ".se-lab/records",
) -> int:
    event_store = EventStore(events_dir)
    artifact_store = ArtifactStore(artifacts_dir)
    record_store = RecordStore(records_dir)
    records = record_run(run_id, event_store, artifact_store, record_store=record_store)
    print(
        json.dumps(
            {
                "status": "PASS",
                "run_id": run_id,
                "record_count": len(records),
                "records_dir": records_dir,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _replay_command(
    run_id: str,
    events_dir: str = ".se-lab/events",
    artifacts_dir: str = ".se-lab/artifacts",
    records_dir: str = ".se-lab/records",
) -> int:
    event_store = EventStore(events_dir)
    artifact_store = ArtifactStore(artifacts_dir)
    record_store = RecordStore(records_dir)
    replay_payload = replay_run(run_id, event_store, artifact_store, record_store=record_store)
    print(json.dumps(replay_payload, indent=2, sort_keys=True))
    return 0 if replay_payload.get("status") == "PASS" else 1


def _evaluate_command(task: str, patch: str) -> int:
    evaluator = IndependentEvaluator()
    result = evaluator.evaluate(task, patch)
    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


def _phase6_command(
    run_id: str,
    events_dir: str = ".se-lab/events",
    artifacts_dir: str = ".se-lab/artifacts",
) -> int:
    audit = evaluate_run(run_id, EventStore(events_dir), ArtifactStore(artifacts_dir))
    print(json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if audit.verdict == "PASS" else 1


def _experiment_command(config_path: str, output: str | None = None) -> int:
    config = ExperimentConfig.model_validate_json(Path(config_path).read_text(encoding="utf-8"))
    result = ExperimentRunner().run(config)
    payload = result.model_dump(mode="json")
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not result.mismatches else 1


def _compare_command(left: str, right: str) -> int:
    left_result = ExperimentResult.model_validate_json(Path(left).read_text(encoding="utf-8"))
    right_result = ExperimentResult.model_validate_json(Path(right).read_text(encoding="utf-8"))
    payload = compare_experiments(left_result, right_result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["comparable"] else 1


def _validate_benchmark(catalog: BenchmarkCatalog) -> dict[str, object]:
    task_ids = [task.task_id for task in catalog.tasks]
    duplicate_ids = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
    task_results: list[dict[str, object]] = []
    for task in catalog.tasks:
        repository = Path(task.repository_path).resolve()
        errors: list[str] = []
        if not (repository / ".git").exists():
            errors.append("repository path is not a git checkout")
        else:
            commit_check = subprocess.run(
                ["git", "-C", str(repository), "cat-file", "-e", f"{task.commit_sha}^{{commit}}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if commit_check.returncode != 0:
                errors.append(f"base commit is unavailable: {task.commit_sha}")
        for relative_path in task.target_tests + task.retained_tests:
            if not (repository / relative_path).exists():
                errors.append(f"test path is missing: {relative_path}")
        if not task.allowed_write_paths:
            errors.append("allowed_write_paths is empty")
        if not task.provenance or not task.deterministic_setup:
            errors.append("provenance or deterministic_setup is missing")
        task_results.append({"task_id": task.task_id, "valid": not errors, "errors": errors})
    if duplicate_ids:
        for task_result in task_results:
            if task_result["task_id"] in duplicate_ids:
                task_result["valid"] = False
                task_result["errors"].append("duplicate task_id")
    return {
        "valid": not duplicate_ids and all(result["valid"] for result in task_results),
        "task_count": len(catalog.tasks),
        "duplicate_task_ids": duplicate_ids,
        "tasks": task_results,
        "validation_note": "Schema, provenance, repository commit, and declared test paths were checked; execution is performed by the smoke and experiment tests.",
    }


def _benchmark_command(catalog: str, action: str | None = None) -> int:
    loaded = BenchmarkCatalog.from_file(catalog)
    if action == "validate":
        payload = _validate_benchmark(loaded)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["valid"] else 1
    payload = {
        "name": loaded.name,
        "version": loaded.version,
        "task_count": len(loaded.tasks),
        "adversarial_count": sum(task.adversarial for task in loaded.tasks),
        "tasks": [task.model_dump(mode="json") for task in loaded.tasks],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _baseline_command(
    task: str,
    *,
    run_id: str | None = None,
    events_dir: str = ".se-lab/events",
    artifacts_dir: str = ".se-lab/artifacts",
    seed: int = 0,
    max_model_calls: int = 1,
    max_wall_clock: int = 300,
    max_retries: int = 0,
    provider_name: str = "mock",
    provider_version: str = "mock-v1",
    model_name: str = "mock-baseline",
) -> int:
    task_obj = EvaluationTask.from_file(task)
    config = BaselineConfig(
        run_id=run_id or "",
        seed=seed,
        max_model_calls=max_model_calls,
        max_wall_clock=max_wall_clock,
        max_retries=max_retries,
        provider_name=provider_name,
        provider_version=provider_version,
        model_name=model_name,
        events_dir=events_dir,
        artifacts_dir=artifacts_dir,
    )
    result = SingleAgentBaseline().run(task_obj, config=config)
    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


def _multi_agent_command(
    task: str,
    *,
    run_id: str | None = None,
    events_dir: str = ".se-lab/events",
    artifacts_dir: str = ".se-lab/artifacts",
    seed: int = 0,
    max_model_calls: int = 4,
    max_tool_calls: int = 10,
    max_wall_clock: int = 300,
    max_retries: int = 1,
    provider_name: str = "mock",
    provider_version: str = "mock-v1",
    model_name: str = "mock-baseline",
) -> int:
    task_obj = EvaluationTask.from_file(task)
    config = MultiAgentConfig(
        run_id=run_id or "",
        seed=seed,
        max_model_calls=max_model_calls,
        max_tool_calls=max_tool_calls,
        max_wall_clock=max_wall_clock,
        max_retries=max_retries,
        provider_name=provider_name,
        provider_version=provider_version,
        model_name=model_name,
        events_dir=events_dir,
        artifacts_dir=artifacts_dir,
    )
    result = MultiAgentWorkflow().run(task_obj, config=config)
    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="se-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-manifest", help="Validate a manifest file")
    validate_parser.add_argument("path")
    validate_parser.set_defaults(handler=_validate_manifest)

    run_parser = subparsers.add_parser("run", help="Create a synthetic run and report")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--events-dir", default=".se-lab/events")
    run_parser.add_argument("--artifacts-dir", default=".se-lab/artifacts")
    run_parser.set_defaults(handler=_run_command)

    report_parser = subparsers.add_parser("report", help="Read or rebuild a run report")
    report_selector = report_parser.add_mutually_exclusive_group(required=True)
    report_selector.add_argument("--run-id")
    report_selector.add_argument("--experiment-result")
    report_parser.add_argument("--events-dir", default=".se-lab/events")
    report_parser.add_argument("--artifacts-dir", default=".se-lab/artifacts")
    report_parser.add_argument("--kind", choices=["experiment", "failure"], default="experiment")
    report_parser.set_defaults(handler=_report_command)

    record_parser = subparsers.add_parser("record", help="Record deterministic observations for a run")
    record_parser.add_argument("--run-id", required=True)
    record_parser.add_argument("--events-dir", default=".se-lab/events")
    record_parser.add_argument("--artifacts-dir", default=".se-lab/artifacts")
    record_parser.add_argument("--records-dir", default=".se-lab/records")
    record_parser.set_defaults(handler=_record_command)

    replay_parser = subparsers.add_parser("replay", help="Replay a run and detect mismatches")
    replay_parser.add_argument("--run-id", required=True)
    replay_parser.add_argument("--events-dir", default=".se-lab/events")
    replay_parser.add_argument("--artifacts-dir", default=".se-lab/artifacts")
    replay_parser.add_argument("--records-dir", default=".se-lab/records")
    replay_parser.set_defaults(handler=_replay_command)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a candidate patch against a deterministic task")
    evaluate_parser.add_argument("--task", required=True)
    evaluate_parser.add_argument("--patch", required=True)
    evaluate_parser.set_defaults(handler=_evaluate_command)

    phase6_parser = subparsers.add_parser("phase6", help="Audit an externally recorded run with Phase 6 metrics")
    phase6_parser.add_argument("--run-id", required=True)
    phase6_parser.add_argument("--events-dir", default=".se-lab/events")
    phase6_parser.add_argument("--artifacts-dir", default=".se-lab/artifacts")
    phase6_parser.set_defaults(handler=_phase6_command)

    experiment_parser = subparsers.add_parser("experiment", help="Run a deterministic matched baseline/treatment experiment")
    experiment_parser.add_argument("--config", required=True)
    experiment_parser.add_argument("--output")
    experiment_parser.set_defaults(handler=_experiment_command)

    compare_parser = subparsers.add_parser("compare", help="Compare two recorded experiment results")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.set_defaults(handler=_compare_command)

    benchmark_parser = subparsers.add_parser("benchmark", help="Inspect a deterministic benchmark task catalog")
    benchmark_parser.add_argument("action", nargs="?", choices=["validate"])
    benchmark_parser.add_argument("--catalog", required=True)
    benchmark_parser.set_defaults(handler=_benchmark_command)

    baseline_parser = subparsers.add_parser("baseline", help="Run the single-agent baseline offline against a deterministic task")
    baseline_parser.add_argument("--task", required=True)
    baseline_parser.add_argument("--run-id")
    baseline_parser.add_argument("--events-dir", default=".se-lab/events")
    baseline_parser.add_argument("--artifacts-dir", default=".se-lab/artifacts")
    baseline_parser.add_argument("--seed", type=int, default=0)
    baseline_parser.add_argument("--max-model-calls", type=int, default=1)
    baseline_parser.add_argument("--max-wall-clock", type=int, default=300)
    baseline_parser.add_argument("--max-retries", type=int, default=0)
    baseline_parser.add_argument("--provider-name", default="mock")
    baseline_parser.add_argument("--provider-version", default="mock-v1")
    baseline_parser.add_argument("--model-name", default="mock-baseline")
    baseline_parser.set_defaults(handler=_baseline_command)

    multi_agent_parser = subparsers.add_parser("multi-agent", help="Run the bounded four-role multi-agent workflow offline")
    multi_agent_parser.add_argument("--task", required=True)
    multi_agent_parser.add_argument("--run-id")
    multi_agent_parser.add_argument("--events-dir", default=".se-lab/events")
    multi_agent_parser.add_argument("--artifacts-dir", default=".se-lab/artifacts")
    multi_agent_parser.add_argument("--seed", type=int, default=0)
    multi_agent_parser.add_argument("--max-model-calls", type=int, default=4)
    multi_agent_parser.add_argument("--max-tool-calls", type=int, default=10)
    multi_agent_parser.add_argument("--max-wall-clock", type=int, default=300)
    multi_agent_parser.add_argument("--max-retries", type=int, default=1)
    multi_agent_parser.add_argument("--provider-name", default="mock")
    multi_agent_parser.add_argument("--provider-version", default="mock-v1")
    multi_agent_parser.add_argument("--model-name", default="mock-baseline")
    multi_agent_parser.set_defaults(handler=_multi_agent_command)

    args = parser.parse_args(argv)
    try:
        if args.command == "validate-manifest":
            return args.handler(args.path)
        if args.command == "run":
            return args.handler(args.manifest, args.events_dir, args.artifacts_dir)
        if args.command == "report":
            return args.handler(args.run_id, args.events_dir, args.artifacts_dir, args.experiment_result, args.kind)
        if args.command == "record":
            return args.handler(args.run_id, args.events_dir, args.artifacts_dir, args.records_dir)
        if args.command == "replay":
            return args.handler(args.run_id, args.events_dir, args.artifacts_dir, args.records_dir)
        if args.command == "evaluate":
            return args.handler(args.task, args.patch)
        if args.command == "phase6":
            return args.handler(args.run_id, args.events_dir, args.artifacts_dir)
        if args.command == "experiment":
            return args.handler(args.config, args.output)
        if args.command == "compare":
            return args.handler(args.left, args.right)
        if args.command == "benchmark":
            return args.handler(args.catalog, args.action)
        if args.command == "baseline":
            return args.handler(
                args.task,
                run_id=args.run_id,
                events_dir=args.events_dir,
                artifacts_dir=args.artifacts_dir,
                seed=args.seed,
                max_model_calls=args.max_model_calls,
                max_wall_clock=args.max_wall_clock,
                max_retries=args.max_retries,
                provider_name=args.provider_name,
                provider_version=args.provider_version,
                model_name=args.model_name,
            )
        if args.command == "multi-agent":
            return args.handler(
                args.task,
                run_id=args.run_id,
                events_dir=args.events_dir,
                artifacts_dir=args.artifacts_dir,
                seed=args.seed,
                max_model_calls=args.max_model_calls,
                max_tool_calls=args.max_tool_calls,
                max_wall_clock=args.max_wall_clock,
                max_retries=args.max_retries,
                provider_name=args.provider_name,
                provider_version=args.provider_version,
                model_name=args.model_name,
            )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    return 1
