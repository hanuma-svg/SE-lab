from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from se_lab.artifacts.store import ArtifactStore
from se_lab.contracts import EventEnvelope
from se_lab.events.store import EventStore
from se_lab.replay.contracts import ReplayFidelity
from se_lab.replay.normalize import compute_request_identity, normalize_request
from se_lab.replay.store import RecordStore


def _event_kind(event: EventEnvelope) -> str:
    if event.event_type == "ModelRequestIssued":
        return "model_request"
    if event.event_type == "ModelResponseReceived":
        return "model_response"
    if event.event_type in {"ToolAuthorizationRequested", "ToolAuthorizationAllowed", "ToolAuthorizationDenied"}:
        return "tool_authorization"
    if event.event_type == "ToolObservationCaptured":
        return "tool_observation"
    return event.event_type


def _extract_artifact_references(payload: dict[str, Any]) -> list[str]:
    refs: list[str] = []

    if isinstance(payload.get("artifact"), dict):
        artifact = payload["artifact"]
        if isinstance(artifact.get("sha256"), str):
            refs.append(f"sha256:{artifact['sha256']}")

    for value in payload.values():
        if isinstance(value, dict):
            refs.extend(_extract_artifact_references(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    refs.extend(_extract_artifact_references(item))
    return refs


def _extract_schema_version(event: EventEnvelope) -> str | None:
    payload = event.payload or {}
    request = payload.get("request") if isinstance(payload.get("request"), dict) else None
    if request is not None and isinstance(request.get("schema_version"), str):
        return request["schema_version"]
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, str):
        return schema_version
    return None


def _build_request_identity(event: EventEnvelope, run_id: str) -> tuple[str, dict[str, Any], str | None, str | None]:
    operation_kind = _event_kind(event)
    payload = event.payload

    normalized_request: dict[str, Any] | None = None
    workspace_snapshot_hash: str | None = None
    policy_version: str | None = None
    provider_context: dict[str, Any] = {
        "provider_name": payload.get("provider_name", "mock"),
        "provider_version": payload.get("provider_version", "mock-v1"),
        "model_name": payload.get("model_name", "mock-baseline"),
        "role": event.role,
    }

    if operation_kind == "tool_authorization":
        normalized_request = normalize_request(payload.get("request"))
        workspace_snapshot_hash = normalized_request.get("workspace_snapshot_hash") if isinstance(normalized_request, dict) else None
        policy_version = payload.get("decision", {}).get("policy_version")
    else:
        normalized_request = normalize_request(payload)

    request_hash = compute_request_identity(
        operation_kind,
        normalized_request,
        schema_version=_extract_schema_version(event),
        workspace_snapshot_hash=workspace_snapshot_hash,
        policy_version=policy_version,
        provider_context=provider_context,
        run_context={"run_id": run_id, "task_id": payload.get("task_id")},
    )
    return request_hash, normalized_request or {}, operation_kind, event.event_id


def record_run(
    run_id: str,
    event_store: EventStore,
    artifact_store: ArtifactStore,
    *,
    record_store: RecordStore | None = None,
) -> list[object]:
    store = record_store or RecordStore(artifact_store.root.parent / "records")
    records: list[object] = []
    previous_event_id: str | None = None

    for event in event_store.read(run_id):
        request_hash, normalized_request, operation_kind, _ = _build_request_identity(event, run_id)
        artifact_references = _extract_artifact_references(event.payload)

        record = store.append_record(
            run_id=run_id,
            observation_payload={
                "event_type": event.event_type,
                "payload": normalize_request(event.payload),
                "artifact_references": artifact_references,
            },
            operation_kind=operation_kind,
            normalized_request=normalized_request,
            request_hash=request_hash,
            causal_parent=previous_event_id,
            sequence=event.sequence,
            schema_version=_extract_schema_version(event),
            workspace_snapshot_hash=(
                event.payload.get("request", {}).get("workspace_snapshot_hash")
                if operation_kind == "tool_authorization"
                else None
            ),
            policy_version=(
                event.payload.get("decision", {}).get("policy_version")
                if operation_kind == "tool_authorization"
                else None
            ),
            provider_context={
                "provider_name": event.payload.get("provider_name", "mock"),
                "provider_version": event.payload.get("provider_version", "mock-v1"),
                "model_name": event.payload.get("model_name", "mock-baseline"),
                "role": event.role,
            },
            artifact_references=artifact_references,
        )

        records.append(record)
        previous_event_id = event.event_id

    return records


def _write_replay_evidence(
    store: RecordStore,
    artifact_store: ArtifactStore,
    run_id: str,
    payload: dict[str, Any],
) -> None:
    evidence_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    evidence_artifact = artifact_store.store_bytes(evidence_bytes, object_type="replay-evidence")

    evidence_path = Path(store.root) / f"{run_id}.replay-evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "artifact_type": "replay-evidence",
                "sha256": evidence_artifact["sha256"],
                "artifact_path": evidence_artifact["path"],
                "content_addressed": True,
                "payload": payload,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def replay_run(
    run_id: str,
    event_store: EventStore,
    artifact_store: ArtifactStore,
    *,
    record_store: RecordStore | None = None,
    fidelity: ReplayFidelity | str = ReplayFidelity.DETERMINISTIC_OBSERVATION,
) -> dict[str, Any]:
    if isinstance(fidelity, str):
        fidelity = ReplayFidelity(fidelity)

    store = record_store or RecordStore(artifact_store.root.parent / "records")
    events = event_store.read(run_id)
    if not events:
        raise FileNotFoundError(f"No events found for run_id={run_id}")

    try:
        recorded = store.load_run(run_id)
    except ValueError as exc:
        failure_payload = {
            "status": "FAIL",
            "run_id": run_id,
            "replay_ok": False,
            "replayed_observations": 0,
            "live_execution_performed": False,
            "fidelity": fidelity.value,
            "mismatch_count": 1,
            "mismatches": [{"reason": "malformed record", "error": str(exc)}],
            "records": [],
            "source_run_id": run_id,
        }
        _write_replay_evidence(store, artifact_store, run_id, failure_payload)
        return failure_payload

    mismatches: list[dict[str, Any]] = []
    for event in events:
        payload = event.payload or {}
        expected_request = payload.get("expected_request")
        actual_request = payload.get("actual_request")
        if expected_request is not None and actual_request is not None:
            normalized_expected = normalize_request(expected_request)
            normalized_actual = normalize_request(actual_request)
            if normalized_expected != normalized_actual:
                mismatches.append(
                    {
                        "reason": "request mismatch",
                        "event_type": event.event_type,
                        "expected_request": normalized_expected,
                        "actual_request": normalized_actual,
                    }
                )

    if not recorded:
        failure_payload = {
            "status": "FAIL",
            "run_id": run_id,
            "replay_ok": False,
            "replayed_observations": 0,
            "live_execution_performed": False,
            "fidelity": fidelity.value,
            "mismatch_count": len(mismatches) or 1,
            "mismatches": mismatches or [{"reason": "No recorded observations found for replay."}],
            "records": [],
            "source_run_id": run_id,
        }
        _write_replay_evidence(store, artifact_store, run_id, failure_payload)
        return failure_payload

    replayed_observations = 0
    previous_event_id: str | None = None

    for record in recorded:
        if not store.validate_record(record):
            mismatches.append(
                {
                    "reason": "integrity check failed",
                    "event_type": record.operation_kind,
                    "request_hash": record.request_hash,
                }
            )

    expectations: dict[str, list[Any]] = {}
    for record in recorded:
        expectations.setdefault(record.request_hash, []).append(record)

    if fidelity == ReplayFidelity.EVENT_EVIDENCE:
        for index, event in enumerate(events):
            observed_record = recorded[index] if index < len(recorded) else None
            if observed_record is None:
                mismatches.append(
                    {
                        "reason": "missing observation",
                        "event_type": event.event_type,
                        "expected_record_index": index,
                    }
                )
                continue

            operation_kind = _event_kind(event)
            if observed_record.operation_kind != operation_kind:
                mismatches.append(
                    {
                        "reason": "operation kind mismatch",
                        "event_type": event.event_type,
                        "expected_operation_kind": observed_record.operation_kind,
                        "actual_operation_kind": operation_kind,
                    }
                )
                continue

            if observed_record.sequence is not None and event.sequence is not None and observed_record.sequence != event.sequence:
                mismatches.append(
                    {
                        "reason": "sequence mismatch",
                        "event_type": event.event_type,
                        "expected_sequence": observed_record.sequence,
                        "actual_sequence": event.sequence,
                    }
                )
                continue

            if observed_record.causal_parent is not None and (
                previous_event_id is None or observed_record.causal_parent != previous_event_id
            ):
                mismatches.append(
                    {
                        "reason": "causal mismatch",
                        "event_type": event.event_type,
                        "expected_causal_parent": observed_record.causal_parent,
                        "actual_causal_parent": previous_event_id,
                    }
                )
                continue

            replayed_observations += 1
            previous_event_id = event.event_id

    else:
        for event in events:
            request_hash, normalized_request, operation_kind, event_id = _build_request_identity(event, run_id)
            observed_records = expectations.get(request_hash, [])

            if not observed_records:
                mismatches.append(
                    {
                        "reason": "missing observation",
                        "event_type": event.event_type,
                        "expected_request_hash": request_hash,
                        "normalized_request": normalized_request,
                    }
                )
                continue

            observed_record = observed_records[0]

            if observed_record.operation_kind != operation_kind:
                mismatches.append(
                    {
                        "reason": "operation kind mismatch",
                        "event_type": event.event_type,
                        "expected_operation_kind": observed_record.operation_kind,
                        "actual_operation_kind": operation_kind,
                    }
                )
                continue

            if observed_record.sequence is not None and event.sequence is not None and observed_record.sequence != event.sequence:
                mismatches.append(
                    {
                        "reason": "sequence mismatch",
                        "event_type": event.event_type,
                        "expected_sequence": observed_record.sequence,
                        "actual_sequence": event.sequence,
                    }
                )
                continue

            if fidelity in {
                ReplayFidelity.CONTEXT_VALIDATION,
                ReplayFidelity.SIDE_EFFECT_FREE,
                ReplayFidelity.RECONSTRUCTION,
            }:
                request_workspace_snapshot_hash = event.payload.get("request", {}).get("workspace_snapshot_hash")
                if (
                    observed_record.workspace_snapshot_hash is not None
                    and request_workspace_snapshot_hash is not None
                    and observed_record.workspace_snapshot_hash != request_workspace_snapshot_hash
                ):
                    mismatches.append(
                        {
                            "reason": "workspace snapshot mismatch",
                            "event_type": event.event_type,
                            "expected": observed_record.workspace_snapshot_hash,
                            "actual": request_workspace_snapshot_hash,
                        }
                    )
                    continue

                decision_policy_version = event.payload.get("decision", {}).get("policy_version")
                if (
                    observed_record.policy_version is not None
                    and decision_policy_version is not None
                    and observed_record.policy_version != decision_policy_version
                ):
                    mismatches.append(
                        {
                            "reason": "policy version mismatch",
                            "event_type": event.event_type,
                            "expected": observed_record.policy_version,
                            "actual": decision_policy_version,
                        }
                    )
                    continue

            if fidelity in {ReplayFidelity.SIDE_EFFECT_FREE, ReplayFidelity.RECONSTRUCTION} and observed_record.artifact_references:
                for reference in observed_record.artifact_references:
                    if reference.startswith("sha256:"):
                        artifact_store.load_bytes(reference.split(":", 1)[1])

            if observed_record.causal_parent is not None and (
                previous_event_id is None or observed_record.causal_parent != previous_event_id
            ):
                mismatches.append(
                    {
                        "reason": "causal mismatch",
                        "event_type": event.event_type,
                        "expected_causal_parent": observed_record.causal_parent,
                        "actual_causal_parent": previous_event_id,
                    }
                )
                continue

            if (
                fidelity == ReplayFidelity.RECONSTRUCTION
                and observed_record.observation_payload != normalize_request(event.payload)
            ):
                mismatches.append(
                    {
                        "reason": "reconstruction payload mismatch",
                        "event_type": event.event_type,
                        "expected_observation_payload": observed_record.observation_payload,
                        "actual_observation_payload": normalize_request(event.payload),
                    }
                )
                continue

            replayed_observations += 1
            previous_event_id = event_id

    payload = {
        "status": "PASS" if not mismatches else "FAIL",
        "run_id": run_id,
        "replay_ok": not mismatches,
        "replayed_observations": replayed_observations,
        "live_execution_performed": False,
        "fidelity": fidelity.value,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "records": [record.model_dump(mode="json") for record in recorded],
        "source_run_id": run_id,
    }
    _write_replay_evidence(store, artifact_store, run_id, payload)
    return payload
