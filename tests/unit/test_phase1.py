from __future__ import annotations

import json

from se_lab.artifacts.store import ArtifactStore
from se_lab.cli import main as cli_main
from se_lab.contracts import (
    EvaluationResult,
    EventEnvelope,
    PolicyDecision,
    RunManifest,
    ToolRequest,
)
from se_lab.events.store import EventStore
from se_lab.replay.replay import replay_run
from se_lab.reporting.report import build_report


def _manifest_data() -> dict[str, object]:
    return {
        "experiment_id": "exp-001",
        "run_id": "run-001",
        "task_id": "task-001",
        "repository_url": "https://example.com/repo.git",
        "repository_commit_sha": "a" * 40,
        "container_image_digest": "sha256:abc123",
        "dependency_lock_hash": "lock-hash",
        "model_provider": "mock-provider",
        "model_version": "v1",
        "adapter_version": "v1.0.0",
        "system_prompt_hash": "sys-hash",
        "role_prompt_hash": "role-hash",
        "tool_schema_version": "tool-schema-v1",
        "policy_version": "policy-v1",
        "temperature": 0.1,
        "sampling_parameters": {"stop": ["\n\n"], "max_tokens": 128},
        "seed": 42,
        "maximum_model_calls": 5,
        "maximum_tool_calls": 10,
        "maximum_tokens": 1024,
        "maximum_wall_clock": 300,
        "maximum_retries": 2,
        "baseline_or_treatment": "baseline",
        "artifact_retention_policy": "retain",
        "redaction_policy": "hash-sensitive",
    }


def test_run_manifest_round_trip(tmp_path):
    manifest = RunManifest(**_manifest_data())
    manifest_path = tmp_path / "manifest.json"
    manifest.write_to_file(manifest_path)

    loaded = RunManifest.read_from_file(manifest_path)

    assert loaded.run_id == manifest.run_id
    assert loaded.model_dump(mode="json") == manifest.model_dump(mode="json")
    assert loaded.content_hash() == manifest.content_hash()


def test_event_store_appends_in_order(tmp_path):
    store = EventStore(tmp_path / "events")

    first = store.append(
        EventEnvelope(
            run_id="run-001",
            event_type="RunCreated",
            role="system",
            payload={"task_id": "task-001"},
        )
    )
    second = store.append(
        EventEnvelope(
            run_id="run-001",
            event_type="WorkspaceCreated",
            role="system",
            payload={"workspace": "demo"},
        )
    )

    events = store.read("run-001")

    assert [event.sequence for event in events] == [1, 2]
    assert first.sequence == 1
    assert second.sequence == 2


def test_artifact_store_records_hashed_artifacts(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")

    artifact = store.store_bytes(b"hello world", object_type="prompt")

    stored_bytes = (tmp_path / "artifacts" / "sha256" / artifact["sha256"][:2] / artifact["sha256"]).read_bytes()

    assert stored_bytes == b"hello world"
    assert artifact["bytes"] == 11
    assert artifact["sha256"] == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_event_store_persists_after_reopening(tmp_path):
    store = EventStore(tmp_path / "events")

    store.append(
        EventEnvelope(
            run_id="run-001",
            event_type="RunCreated",
            role="system",
            payload={"task_id": "task-001"},
        )
    )

    reopened = EventStore(tmp_path / "events")
    events = reopened.read("run-001")

    assert len(events) == 1
    assert events[0].event_type == "RunCreated"


def test_artifact_store_reuses_existing_artifacts(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")

    first = store.store_bytes(b"payload", object_type="patch")
    second = store.store_bytes(b"payload", object_type="patch")

    assert first["sha256"] == second["sha256"]
    assert len(store.list()) == 1


def test_build_report_includes_events_and_artifacts(tmp_path):
    event_store = EventStore(tmp_path / "events")
    artifact_store = ArtifactStore(tmp_path / "artifacts")

    event_store.append(
        EventEnvelope(
            run_id="run-001",
            event_type="RunCreated",
            role="system",
            payload={"task_id": "task-001"},
        )
    )
    artifact_store.store_bytes(b"payload", object_type="patch")

    report = build_report("run-001", event_store, artifact_store)

    assert report["run_id"] == "run-001"
    assert report["event_count"] == 1
    assert report["artifacts"][0]["object_type"] == "patch"


def test_cli_validate_manifest(tmp_path, capsys):
    manifest = RunManifest(**_manifest_data())
    manifest_path = tmp_path / "manifest.json"
    manifest.write_to_file(manifest_path)

    exit_code = cli_main(["validate-manifest", str(manifest_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["valid"] is True
    assert payload["run_id"] == "run-001"


def test_cli_reports_invalid_manifest(tmp_path, capsys):
    manifest_path = tmp_path / "invalid.json"
    manifest_path.write_text('{"not": "a manifest"}\n', encoding="utf-8")

    exit_code = cli_main(["validate-manifest", str(manifest_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(captured.out)
    assert payload["valid"] is False
    assert "error" in payload


def test_cli_run_and_report_and_replay(tmp_path):
    manifest = RunManifest(**_manifest_data())
    manifest_path = tmp_path / "manifest.json"
    manifest.write_to_file(manifest_path)

    events_dir = tmp_path / "events"
    artifacts_dir = tmp_path / "artifacts"
    records_dir = tmp_path / "records"

    cli_main(["run", "--manifest", str(manifest_path), "--events-dir", str(events_dir), "--artifacts-dir", str(artifacts_dir)])

    report_code = cli_main(["report", "--run-id", "run-001", "--events-dir", str(events_dir), "--artifacts-dir", str(artifacts_dir)])
    record_code = cli_main(
        [
            "record",
            "--run-id",
            "run-001",
            "--events-dir",
            str(events_dir),
            "--artifacts-dir",
            str(artifacts_dir),
            "--records-dir",
            str(records_dir),
        ]
    )
    replay_code = cli_main(
        [
            "replay",
            "--run-id",
            "run-001",
            "--events-dir",
            str(events_dir),
            "--artifacts-dir",
            str(artifacts_dir),
            "--records-dir",
            str(records_dir),
        ]
    )

    assert report_code == 0
    assert record_code == 0
    assert replay_code == 0


def test_replay_detects_mismatch(tmp_path):
    event_store = EventStore(tmp_path / "events")
    artifact_store = ArtifactStore(tmp_path / "artifacts")

    event_store.append(
        EventEnvelope(
            run_id="run-001",
            event_type="ToolAuthorizationRequested",
            role="planner",
            payload={
                "expected_request": {"tool_name": "read_file", "arguments": {"path": "README.md"}},
                "actual_request": {"tool_name": "read_file", "arguments": {"path": "src"}},
            },
        )
    )

    replay_payload = replay_run("run-001", event_store, artifact_store)

    assert replay_payload["replay_ok"] is False
    assert replay_payload["mismatch_count"] == 1
    assert replay_payload["mismatches"][0]["expected_request"] == {"tool_name": "read_file", "arguments": {"path": "README.md"}}


def test_contract_models_are_useful():
    tool_request = ToolRequest(
        tool_name="read_file",
        arguments={"path": "README.md"},
        workspace_snapshot_hash="abc123",
        schema_version="schema-v1",
    )
    policy_decision = PolicyDecision(
        decision="deny",
        capability="READ_ENV",
        reason="Environment variables are denied by default.",
    )
    evaluation_result = EvaluationResult(
        status="pass",
        summary="Target tests passed.",
        pass_to_fail_tests=3,
        retained_tests=5,
    )

    assert tool_request.tool_name == "read_file"
    assert policy_decision.decision == "deny"
    assert evaluation_result.status == "pass"
