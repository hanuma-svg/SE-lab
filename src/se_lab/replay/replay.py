from __future__ import annotations

import json
from typing import Any

from se_lab.artifacts.store import ArtifactStore
from se_lab.events.store import EventStore


def normalize_request(request: dict[str, Any] | None) -> dict[str, Any] | None:
    if request is None:
        return None
    return json.loads(json.dumps(request, sort_keys=True))


def detect_mismatch(
    expected_request: dict[str, Any] | None,
    actual_request: dict[str, Any] | None,
    *,
    role: str | None = None,
    event_sequence: int | None = None,
) -> dict[str, Any] | None:
    if normalize_request(expected_request) == normalize_request(actual_request):
        return None

    return {
        "first_divergence": "request",
        "expected_request": normalize_request(expected_request),
        "actual_request": normalize_request(actual_request),
        "role": role,
        "event_sequence": event_sequence,
        "artifact_references": [],
    }


def replay_run(run_id: str, event_store: EventStore, artifact_store: ArtifactStore) -> dict[str, Any]:
    events = event_store.read(run_id)
    if not events:
        raise FileNotFoundError(f"No events found for run_id={run_id}")

    sequence_mismatches: list[str] = []
    mismatches: list[dict[str, Any]] = []

    previous_sequence = 0
    for event in events:
        current_sequence = event.sequence or 0
        if current_sequence != previous_sequence + 1:
            sequence_mismatches.append(
                f"Expected sequence {previous_sequence + 1}, found {current_sequence} for event_id={event.event_id}"
            )
        previous_sequence = current_sequence

        expected_request = event.payload.get("expected_request")
        actual_request = event.payload.get("actual_request")
        mismatch = detect_mismatch(
            expected_request,
            actual_request,
            role=event.role,
            event_sequence=event.sequence,
        )
        if mismatch is not None:
            mismatches.append(mismatch)

        for value in event.payload.values():
            if isinstance(value, str) and value.startswith("sha256:"):
                artifact_store.load_bytes(value.replace("sha256:", "", 1))

    return {
        "run_id": run_id,
        "replay_ok": not mismatches and not sequence_mismatches,
        "events": [event.model_dump(mode="json") for event in events],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "sequence_mismatches": sequence_mismatches,
    }
