from __future__ import annotations

import argparse
import json
from pathlib import Path

from se_lab.artifacts.store import ArtifactStore
from se_lab.contracts import EventEnvelope, RunManifest
from se_lab.evaluation import IndependentEvaluator
from se_lab.events.store import EventStore
from se_lab.replay.replay import replay_run
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


def _replay_command(run_id: str, events_dir: str = ".se-lab/events", artifacts_dir: str = ".se-lab/artifacts") -> int:
    event_store = EventStore(events_dir)
    artifact_store = ArtifactStore(artifacts_dir)
    replay_payload = replay_run(run_id, event_store, artifact_store)
    print(json.dumps(replay_payload, indent=2, sort_keys=True))
    return 0


def _evaluate_command(task: str, patch: str) -> int:
    evaluator = IndependentEvaluator()
    result = evaluator.evaluate(task, patch)
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

    replay_parser = subparsers.add_parser("replay", help="Replay a run and detect mismatches")
    replay_parser.add_argument("--run-id", required=True)
    replay_parser.add_argument("--events-dir", default=".se-lab/events")
    replay_parser.add_argument("--artifacts-dir", default=".se-lab/artifacts")
    replay_parser.set_defaults(handler=_replay_command)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a candidate patch against a deterministic task")
    evaluate_parser.add_argument("--task", required=True)
    evaluate_parser.add_argument("--patch", required=True)
    evaluate_parser.set_defaults(handler=_evaluate_command)

    args = parser.parse_args(argv)
    try:
        if args.command == "validate-manifest":
            return args.handler(args.path)
        if args.command == "run":
            return args.handler(args.manifest, args.events_dir, args.artifacts_dir)
        if args.command == "report":
            return args.handler(args.run_id, args.events_dir, args.artifacts_dir)
        if args.command == "replay":
            return args.handler(args.run_id, args.events_dir, args.artifacts_dir)
        if args.command == "evaluate":
            return args.handler(args.task, args.patch)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    return 1
