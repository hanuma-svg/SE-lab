from __future__ import annotations

import json

from se_lab.artifacts.store import ArtifactStore
from se_lab.cli import main as cli_main
from se_lab.contracts import EventEnvelope
from se_lab.events.store import EventStore
from se_lab.replay.contracts import ReplayFidelity
from se_lab.replay.normalize import compute_request_identity, normalize_request
from se_lab.replay.replay import record_run, replay_run
from se_lab.replay.store import RecordStore


def test_request_normalization_is_deterministic_and_context_sensitive():
    request = {
        "tool_name": "read_file",
        "arguments": {"path": "README.md", "mode": "r"},
        "workspace_snapshot_hash": "abc123",
        "schema_version": "v1",
        "volatile": {"event_id": "should-not-matter", "timestamp": "2026-01-01T00:00:00Z"},
    }

    first = compute_request_identity(
        "tool_request",
        request,
        schema_version="v1",
        workspace_snapshot_hash="abc123",
        policy_version="policy-v1",
        provider_context={"role": "planner", "model_name": "mock"},
    )
    second = compute_request_identity(
        "tool_request",
        {
            "tool_name": "read_file",
            "arguments": {"mode": "r", "path": "README.md"},
            "schema_version": "v1",
            "workspace_snapshot_hash": "abc123",
            "policy_version": "policy-v1",
            "volatile": {"timestamp": "2026-01-02T00:00:00Z", "event_id": "different"},
        },
        schema_version="v1",
        workspace_snapshot_hash="abc123",
        policy_version="policy-v1",
        provider_context={"model_name": "mock", "role": "planner"},
    )

    changed = compute_request_identity(
        "tool_request",
        request,
        schema_version="v2",
        workspace_snapshot_hash="abc123",
        policy_version="policy-v1",
        provider_context={"role": "planner", "model_name": "mock"},
    )

    assert first == second
    assert normalize_request(request) == normalize_request({
        "volatile": {"event_id": "different", "timestamp": "2026-01-02T00:00:00Z"},
        "arguments": {"path": "README.md", "mode": "r"},
        "tool_name": "read_file",
        "workspace_snapshot_hash": "abc123",
        "schema_version": "v1",
    })
    assert first != changed


def test_record_store_appends_observations_and_preserves_integrity(tmp_path):
    store = RecordStore(tmp_path / "records")

    record = store.append_record(
        run_id="run-001",
        observation_payload={"status": "ok"},
        operation_kind="tool_observation",
        normalized_request={"tool_name": "read_file", "arguments": {"path": "README.md"}},
        request_hash="request-abc",
        causal_parent=None,
        sequence=1,
        schema_version="phase-5",
        workspace_snapshot_hash="ws-hash",
        policy_version="policy-v1",
        provider_context={"role": "planner"},
        artifact_references=["sha256:abc"],
    )

    loaded = store.lookup_request(record.request_hash)

    assert loaded.record_id == record.record_id
    assert loaded.integrity_hash == record.integrity_hash
    assert store.validate_record(loaded) is True


def test_record_run_and_replay_are_side_effect_free_and_well_formed(tmp_path):
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
                "repository_path": "/tmp/repo",
                "commit_sha": "deadbeef",
                "provider_name": "mock",
                "provider_version": "mock-v1",
                "model_name": "mock-baseline",
                "metadata": {"mock_patch": "abc"},
            },
        )
    )
    event_store.append(
        EventEnvelope(
            run_id="run-001",
            event_type="ModelResponseReceived",
            role="planner",
            payload={
                "provider_name": "mock",
                "provider_version": "mock-v1",
                "model_name": "mock-baseline",
                "artifact": {"sha256": "abc", "object_type": "model_response"},
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "finish_reason": "stop",
            },
        )
    )

    record_store = RecordStore(records_dir)
    records = record_run("run-001", event_store, artifact_store, record_store=record_store)

    replay_payload = replay_run(
        "run-001",
        event_store,
        artifact_store,
        record_store=record_store,
        fidelity=ReplayFidelity.DETERMINISTIC_OBSERVATION,
    )

    assert replay_payload["status"] == "PASS"
    assert replay_payload["replayed_observations"] == len(records)
    assert replay_payload["live_execution_performed"] is False
    assert records[0].operation_kind == "model_request"


def test_replay_rejects_mismatched_request_identity_and_missing_observation(tmp_path):
    events_dir = tmp_path / "events"
    artifacts_dir = tmp_path / "artifacts"
    records_dir = tmp_path / "records"

    event_store = EventStore(events_dir)
    artifact_store = ArtifactStore(artifacts_dir)
    event_store.append(
        EventEnvelope(
            run_id="run-001",
            event_type="ToolAuthorizationRequested",
            role="policy_gateway",
            payload={
                "request": {
                    "tool_name": "read_file",
                    "arguments": {"path": "README.md"},
                    "workspace_snapshot_hash": "ws-hash",
                    "schema_version": "phase-5",
                },
                "decision": {
                    "decision": "allow",
                    "capability": "READ_REPOSITORY",
                    "reason": "safe",
                    "request_id": "request-1",
                    "policy_version": "policy-v1",
                },
            },
        )
    )

    record_store = RecordStore(records_dir)
    record_run("run-001", event_store, artifact_store, record_store=record_store)

    replay_payload = replay_run("run-001", event_store, artifact_store, record_store=record_store)
    assert replay_payload["status"] == "PASS"

    bad_record = record_store.load_run("run-001")[0]
    bad_record.request_hash = "tampered-hash"
    record_store._write_record(bad_record)

    replay_payload = replay_run("run-001", event_store, artifact_store, record_store=record_store)
    assert replay_payload["status"] == "FAIL"
    assert replay_payload["mismatches"]

    try:
        record_store.lookup_request("missing-request-hash")
    except KeyError:
        pass
    else:
        raise AssertionError("lookup_request should fail closed for missing observations")


def test_cli_record_and_replay_commands_round_trip(tmp_path, capsys):
    events_dir = tmp_path / "events"
    artifacts_dir = tmp_path / "artifacts"
    records_dir = tmp_path / "records"

    event_store = EventStore(events_dir)
    event_store.append(
        EventEnvelope(
            run_id="run-001",
            event_type="ModelRequestIssued",
            role="planner",
            payload={
                "task_id": "task-001",
                "repository_path": "/tmp/repo",
                "commit_sha": "deadbeef",
                "provider_name": "mock",
                "provider_version": "mock-v1",
                "model_name": "mock-baseline",
                "metadata": {"mock_patch": "abc"},
            },
        )
    )
    event_store.append(
        EventEnvelope(
            run_id="run-001",
            event_type="ModelResponseReceived",
            role="planner",
            payload={
                "provider_name": "mock",
                "provider_version": "mock-v1",
                "model_name": "mock-baseline",
                "artifact": {"sha256": "abc", "object_type": "model_response"},
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                "finish_reason": "stop",
            },
        )
    )

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
    record_output = capsys.readouterr().out.strip()
    assert record_code == 0
    assert json.loads(record_output)["status"] == "PASS"

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
    replay_output = capsys.readouterr().out.strip()

    assert replay_code == 0
    payload = json.loads(replay_output)
    assert payload["status"] == "PASS"
