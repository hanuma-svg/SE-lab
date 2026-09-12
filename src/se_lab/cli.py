from __future__ import annotations

import argparse
import json
from pathlib import Path

from se_lab.agents import (
    BaselineConfig,
    MultiAgentConfig,
    MultiAgentWorkflow,
    SingleAgentBaseline,
)
from se_lab.artifacts.store import ArtifactStore
from se_lab.contracts import EventEnvelope, RunManifest
from se_lab.evaluation import EvaluationTask, IndependentEvaluator
from se_lab.events.store import EventStore
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


def _report_command(run_id: str, events_dir: str = ".se-lab/events", artifacts_dir: str = ".se-lab/artifacts") -> int:
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
    report_parser.add_argument("--run-id", required=True)
    report_parser.add_argument("--events-dir", default=".se-lab/events")
    report_parser.add_argument("--artifacts-dir", default=".se-lab/artifacts")
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
            return args.handler(args.run_id, args.events_dir, args.artifacts_dir)
        if args.command == "record":
            return args.handler(args.run_id, args.events_dir, args.artifacts_dir, args.records_dir)
        if args.command == "replay":
            return args.handler(args.run_id, args.events_dir, args.artifacts_dir, args.records_dir)
        if args.command == "evaluate":
            return args.handler(args.task, args.patch)
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
