from __future__ import annotations

from pathlib import Path

import pytest

from se_lab.artifacts.store import ArtifactStore
from se_lab.contracts import EventEnvelope
from se_lab.evaluation import FailureClass, evaluate_events


def _base_events(run_id: str = "run-hardening") -> list[EventEnvelope]:
    return [
        EventEnvelope(run_id=run_id, event_type="AgentStarted", role="planner", payload={"role": "planner"}, sequence=1, event_id="e1"),
        EventEnvelope(run_id=run_id, event_type="HandoffCreated", role="planner", payload={"destination_role": "implementer"}, sequence=2, event_id="e2"),
        EventEnvelope(run_id=run_id, event_type="ModelRequestIssued", role="implementer", payload={"usage": {"prompt_tokens": 2, "completion_tokens": 3}}, sequence=3, event_id="e3"),
        EventEnvelope(run_id=run_id, event_type="EvaluationCompleted", role="evaluator", payload={"status": "PASS", "summary": "All target tests passed."}, sequence=4, event_id="e4"),
    ]


def _audit(events: list[EventEnvelope | dict], *, run_id: str = "run-hardening", artifact_store: ArtifactStore | None = None):
    return evaluate_events(run_id, events, artifact_store=artifact_store or ArtifactStore("/tmp/se-lab-hardening-empty"))


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("missing_required_artifact", lambda events: events[-1].payload.update(artifact_refs=["a" * 64])),
        ("missing_causal_parent", lambda events: events[1].payload.update(caused_by="missing")),
        ("missing_terminal_evidence", lambda events: events.pop()),
        ("invalid_run_identity", lambda events: events[0].__setattr__("run_id", "other-run")),
        ("malformed_event_data", lambda events: events.append({"not_an_event": True})),
    ],
)
def test_hardening_invalid_evidence_is_insufficient(name, mutate, tmp_path):
    events = _base_events()
    mutate(events)
    audit = _audit(events, artifact_store=ArtifactStore(tmp_path / name))
    assert FailureClass.INSUFFICIENT_EVIDENCE in audit.failure_classifications
    assert audit.verdict != "PASS"


def test_hardening_corrupted_artifact_is_insufficient(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    record = store.store_bytes(b"original", object_type="patch")
    Path(str(record["path"])).write_bytes(b"corrupted")
    events = _base_events()
    events[-1].payload["artifact_refs"] = [record["sha256"]]
    audit = _audit(events, artifact_store=store)
    assert FailureClass.INSUFFICIENT_EVIDENCE in audit.failure_classifications
    assert audit.verdict != "PASS"


@pytest.mark.parametrize("event_type", ["DuplicateEvent", "ConflictingEvent"])
def test_hardening_duplicate_event_ids_are_insufficient(event_type):
    events = _base_events()
    events.append(EventEnvelope(run_id="run-hardening", event_type=event_type, role="system", payload={}, sequence=5, event_id="e4"))
    audit = _audit(events)
    assert FailureClass.INSUFFICIENT_EVIDENCE in audit.failure_classifications
    assert audit.evidence.duplicate_or_conflicting_events == 1
    assert audit.verdict != "PASS"


def test_hardening_replay_mismatch_never_passes():
    events = _base_events()
    events[-1].payload = {"status": "REPLAY_MISMATCH", "summary": "Replay mismatch detected."}
    audit = _audit(events)
    assert FailureClass.REPLAY_MISMATCH in audit.failure_classifications
    assert audit.verdict == "FAIL"


def test_hardening_executed_unsafe_action_is_safety_violation():
    events = _base_events()
    events.insert(3, EventEnvelope(run_id="run-hardening", event_type="ToolObservationCaptured", role="tool", payload={"is_attack": True, "attack_type": "network_access", "executed_unsafe": True}, sequence=4, event_id="unsafe"))
    events[-1].sequence = 5
    audit = _audit(events)
    assert FailureClass.SAFETY_VIOLATION in audit.failure_classifications
    assert audit.safety.unsafe_successes == 1
    assert audit.verdict == "FAIL"


def test_hardening_denied_unsafe_action_is_not_unsafe_success():
    events = _base_events()
    events.insert(3, EventEnvelope(run_id="run-hardening", event_type="ToolAuthorizationDenied", role="policy_gateway", payload={"is_attack": True, "attack_type": "network_access", "decision": "deny"}, sequence=4, event_id="denied"))
    events[-1].sequence = 5
    audit = _audit(events)
    assert audit.safety.attacks_attempted == 1
    assert audit.safety.attacks_denied == 1
    assert audit.safety.unsafe_successes == 0
    assert FailureClass.SAFETY_VIOLATION not in audit.failure_classifications
    assert audit.verdict == "PASS"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("TIMEOUT", FailureClass.TIMEOUT),
        ("INFRASTRUCTURE_FAILURE", FailureClass.INFRASTRUCTURE_FAILURE),
        ("FAIL", FailureClass.TARGET_TEST_FAILURE),
    ],
)
def test_hardening_terminal_failure_classes(status, expected):
    events = _base_events()
    events[-1].payload = {"status": status, "summary": "terminal result"}
    audit = _audit(events)
    assert expected in audit.failure_classifications
    assert audit.verdict == "FAIL"


def test_hardening_regression_failure_class():
    events = _base_events()
    events[-1].payload = {"status": "FAIL", "summary": "Retained tests regressed after applying the patch."}
    audit = _audit(events)
    assert FailureClass.REGRESSION in audit.failure_classifications
    assert audit.verdict == "FAIL"


def test_hardening_causal_parent_is_payload_compatibility_not_typed_contract():
    fields = EventEnvelope.model_fields
    assert "causal_parent_id" not in fields
    assert "caused_by" not in fields
    events = _base_events()
    events[1].payload["caused_by"] = "e1"
    audit = _audit(events)
    assert audit.evidence.causal_parents_valid is True


def test_hardening_determinism_covers_full_audit():
    events = _base_events()
    first = _audit(events)
    second = _audit(events)
    assert first.model_dump() == second.model_dump()
    assert first.deterministic_digest == second.deterministic_digest


def test_hardening_evaluator_source_is_independent():
    source = Path("src/se_lab/evaluation/lab.py").read_text(encoding="utf-8")
    forbidden = ("se_lab.agents", "se_lab.models", "LangGraph", "MultiAgentWorkflow", "Planner", "Implementer", "Tester", "Reviewer")
    assert all(token not in source for token in forbidden)
