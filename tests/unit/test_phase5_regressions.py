from __future__ import annotations

from pathlib import Path

import pytest

from se_lab.artifacts.store import ArtifactStore
from se_lab.cli import main as cli_main
from se_lab.contracts import EventEnvelope
from se_lab.events.store import EventStore
from se_lab.replay.normalize import compute_request_identity
from se_lab.replay.replay import record_run, replay_run
from se_lab.replay.store import RecordStore


def test_request_identity_is_stable_across_runs_and_sensitive_to_context():
    request = {
        "tool_name": "read_file",
        "arguments": {"path": "README.md", "mode": "r"},
        "workspace_snapshot_hash": "ws-1",
        "schema_version": "v1",
    }

    first = compute_request_identity(
        "tool_request",
        request,
        schema_version="v1",
        workspace_snapshot_hash="ws-1",
        policy_version="policy-v1",
        provider_context={"role": "planner", "provider_name": "mock", "provider_version": "mock-v1", "model_name": "mock-baseline"},
        run_context={"run_id": "run-A"},
    )
    second = compute_request_identity(
        "tool_request",
        request,
        schema_version="v1",
        workspace_snapshot_hash="ws-1",
        policy_version="policy-v1",
        provider_context={"role": "planner", "provider_name": "mock", "provider_version": "mock-v1", "model_name": "mock-baseline"},
        run_context={"run_id": "run-B"},
    )

    assert first == second

    payload_changed = compute_request_identity(
        "tool_request",
        {"tool_name": "read_file", "arguments": {"path": "src", "mode": "r"}, "workspace_snapshot_hash": "ws-1", "schema_version": "v1"},
        schema_version="v1",
        workspace_snapshot_hash="ws-1",
        policy_version="policy-v1",
        provider_context={"role": "planner", "provider_name": "mock", "provider_version": "mock-v1", "model_name": "mock-baseline"},
    )
    schema_changed = compute_request_identity(
        "tool_request",
        request,
        schema_version="v2",
        workspace_snapshot_hash="ws-1",
        policy_version="policy-v1",
        provider_context={"role": "planner", "provider_name": "mock", "provider_version": "mock-v1", "model_name": "mock-baseline"},
    )
    workspace_changed = compute_request_identity(
        "tool_request",
        request,
        schema_version="v1",
        workspace_snapshot_hash="ws-2",
        policy_version="policy-v1",
        provider_context={"role": "planner", "provider_name": "mock", "provider_version": "mock-v1", "model_name": "mock-baseline"},
    )
    provider_changed = compute_request_identity(
        "tool_request",
        request,
        schema_version="v1",
        workspace_snapshot_hash="ws-1",
        policy_version="policy-v1",
        provider_context={"role": "planner", "provider_name": "mock", "provider_version": "mock-v1", "model_name": "different-model"},
    )

    assert first != payload_changed
    assert first != schema_changed
    assert first != workspace_changed
    assert first != provider_changed


def test_artifact_store_verifies_sha256_on_load(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")

    stored = store.store_bytes(b"hello", "payload")
    assert store.load_bytes(stored["sha256"]) == b"hello"

    artifact_path = Path(stored["path"])
    artifact_path.write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="integrity"):
        store.load_bytes(stored["sha256"])

    replacement = store.store_bytes(b"goodbye", "payload")
    artifact_path.write_bytes(Path(replacement["path"]).read_bytes())

    with pytest.raises(ValueError, match="integrity"):
        store.load_bytes(stored["sha256"])

    with pytest.raises(FileNotFoundError):
        store.load_bytes("a" * 64)


def test_replay_rejects_malformed_records_and_writes_evidence(tmp_path):
    events_dir = tmp_path / "events"
    artifacts_dir = tmp_path / "artifacts"
    records_dir = tmp_path / "records"

    event_store = EventStore(events_dir)
    artifact_store = ArtifactStore(artifacts_dir)
    event_store.append(
        EventEnvelope(
            run_id="run-001",
            event_type="ModelRequestIssued",
            role="planner",
            payload={
                "task_id": "task-001",
                "provider_name": "mock",
                "provider_version": "mock-v1",
                "model_name": "mock-baseline",
                "metadata": {"mock_patch": "abc"},
            },
        )
    )

    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / "run-001.jsonl").write_text('{"not": "valid"\n', encoding="utf-8")

    store = RecordStore(records_dir)
    replay_payload = replay_run("run-001", event_store, artifact_store, record_store=store)

    assert replay_payload["status"] == "FAIL"
    assert replay_payload["replay_ok"] is False
    assert replay_payload["mismatches"][0]["reason"] == "malformed record"
    assert (records_dir / "run-001.replay-evidence.json").exists()


def test_record_store_accepts_exact_duplicates_and_rejects_conflicts(tmp_path):
    store = RecordStore(tmp_path / "records")

    base_kwargs = {
        "run_id": "run-001",
        "operation_kind": "tool_observation",
        "normalized_request": {"tool_name": "read_file", "arguments": {"path": "README.md"}},
        "request_hash": "request-abc",
        "causal_parent": None,
        "sequence": 1,
        "schema_version": "v1",
        "workspace_snapshot_hash": "ws-1",
        "policy_version": "policy-v1",
        "provider_context": {"role": "planner"},
        "artifact_references": [],
    }

    store.append_record(observation_payload={"status": "ok"}, **base_kwargs)
    store.append_record(observation_payload={"status": "ok"}, **base_kwargs)

    loaded = store.load_run("run-001")
    assert len(loaded) == 1

    conflicting_store = RecordStore(tmp_path / "records-conflict")
    conflicting_store.append_record(observation_payload={"status": "ok"}, **base_kwargs)
    conflicting_store.append_record(
        observation_payload={"status": "conflict"},
        **base_kwargs,
    )

    with pytest.raises(ValueError, match="Conflicting duplicate"):
        conflicting_store.load_run("run-001")


def test_cli_replay_exit_codes(tmp_path):
    events_dir = tmp_path / "events"
    artifacts_dir = tmp_path / "artifacts"
    records_dir = tmp_path / "records"

    event_store = EventStore(events_dir)
    artifact_store = ArtifactStore(artifacts_dir)

    event_store.append(
        EventEnvelope(
            run_id="run-001",
            event_type="ModelRequestIssued",
            role="planner",
            payload={
                "task_id": "task-001",
                "provider_name": "mock",
                "provider_version": "mock-v1",
                "model_name": "mock-baseline",
                "metadata": {"mock_patch": "abc"},
            },
        )
    )

    missing_record_code = cli_main(
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
    assert missing_record_code != 0

    record_run("run-001", event_store, artifact_store, record_store=RecordStore(records_dir))
    success_code = cli_main(
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
    assert success_code == 0


def test_replay_rejects_schema_drift_in_request_identity(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    records_dir = tmp_path / "records"
    v1_events_dir = tmp_path / "events-v1"
    v2_events_dir = tmp_path / "events-v2"

    artifact_store = ArtifactStore(artifacts_dir)
    record_store = RecordStore(records_dir)

    request_v1 = {
        "tool_name": "read_file",
        "arguments": {"path": "README.md"},
        "schema_version": "v1",
        "workspace_snapshot_hash": "ws-1",
    }
    request_v2 = {
        "tool_name": "read_file",
        "arguments": {"path": "README.md"},
        "schema_version": "v2",
        "workspace_snapshot_hash": "ws-1",
    }

    v1_event_store = EventStore(v1_events_dir)
    v2_event_store = EventStore(v2_events_dir)

    v1_event_store.append(
        EventEnvelope(
            run_id="run-001",
            event_type="ToolAuthorizationRequested",
            role="policy_gateway",
            payload={
                "request": request_v1,
                "decision": {
                    "decision": "allow",
                    "capability": "READ_REPOSITORY",
                    "reason": "safe",
                    "policy_version": "policy-v1",
                },
            },
        )
    )

    records = record_run("run-001", v1_event_store, artifact_store, record_store=record_store)
    assert len(records) == 1
    assert records[0].schema_version == "v1"

    replay_payload = replay_run("run-001", v1_event_store, artifact_store, record_store=record_store)
    assert replay_payload["status"] == "PASS"

    v2_event_store.append(
        EventEnvelope(
            run_id="run-001",
            event_type="ToolAuthorizationRequested",
            role="policy_gateway",
            payload={
                "request": request_v2,
                "decision": {
                    "decision": "allow",
                    "capability": "READ_REPOSITORY",
                    "reason": "safe",
                    "policy_version": "policy-v1",
                },
            },
        )
    )

    replay_payload = replay_run("run-001", v2_event_store, artifact_store, record_store=record_store)
    assert replay_payload["status"] == "FAIL"
    assert replay_payload["mismatch_count"] >= 1
    assert any(mismatch.get("normalized_request", {}).get("schema_version") == "v2" for mismatch in replay_payload["mismatches"])
