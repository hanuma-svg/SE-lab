from __future__ import annotations

import json

from se_lab.artifacts.store import ArtifactStore
from se_lab.cli import main as cli_main
from se_lab.contracts import EventEnvelope
from se_lab.evaluation import FailureClass, evaluate_events, evaluate_run
from se_lab.events.store import EventStore


def _events(run_id: str, *, unsafe: bool = False) -> list[EventEnvelope]:
    return [
        EventEnvelope(run_id=run_id, event_type="AgentStarted", role="planner", payload={"role": "planner"}, sequence=1, event_id="e1"),
        EventEnvelope(run_id=run_id, event_type="HandoffCreated", role="planner", payload={"destination_role": "implementer"}, sequence=2, event_id="e2"),
        EventEnvelope(run_id=run_id, event_type="ModelRequestIssued", role="implementer", payload={"usage": {"prompt_tokens": 3, "completion_tokens": 2}}, sequence=3, event_id="e3"),
        EventEnvelope(run_id=run_id, event_type="ToolAuthorizationDenied", role="policy_gateway", payload={"is_attack": True, "attack_type": "path_traversal", "decision": "deny"}, sequence=4, event_id="e4"),
        EventEnvelope(run_id=run_id, event_type="EvaluationCompleted", role="evaluator", payload={"status": "PASS", "summary": "All target tests passed."}, sequence=5, event_id="e5"),
    ]


def test_phase6_measures_independent_run_evidence_and_safety(tmp_path):
    run_id = "run-006"
    events = _events(run_id)
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    record = artifact_store.store_bytes(b"patch", object_type="patch")
    events[-1].payload["artifact_refs"] = [record["sha256"]]

    audit = evaluate_events(run_id, events, artifact_store=artifact_store, budgets={"model_calls": 2})

    assert audit.verdict == "PASS"
    assert audit.primary_failure == FailureClass.SUCCESS
    assert audit.quality.pass_at_1 is True
    assert audit.safety.attacks_attempted == 1
    assert audit.safety.attacks_denied == 1
    assert audit.safety.unsafe_successes == 0
    assert audit.evidence.sha_integrity_valid is True
    assert audit.efficiency.model_calls == 1
    assert audit.efficiency.tokens_captured == 5
    assert audit.efficiency.budget_utilization["model_calls"] == 0.5


def test_phase6_fails_closed_on_missing_causal_parent_and_artifact(tmp_path):
    events = _events("run-006")
    events[1].payload["caused_by"] = "missing"
    events[-1].payload["artifact_refs"] = ["0" * 64]

    audit = evaluate_events("run-006", events, artifact_store=ArtifactStore(tmp_path / "artifacts"))

    assert audit.verdict == "PASS WITH LIMITATIONS"
    assert FailureClass.INSUFFICIENT_EVIDENCE in audit.failure_classifications
    assert audit.evidence.missing_parents == 1
    assert audit.evidence.sha_integrity_valid is False


def test_phase6_is_deterministic_and_cli_audits_stored_run(tmp_path, capsys):
    run_id = "run-006"
    events_dir = tmp_path / "events"
    artifacts_dir = tmp_path / "artifacts"
    store = EventStore(events_dir)
    for event in _events(run_id):
        store.append(event)

    first = evaluate_run(run_id, store, ArtifactStore(artifacts_dir))
    second = evaluate_run(run_id, store, ArtifactStore(artifacts_dir))
    assert first.deterministic_digest == second.deterministic_digest

    assert cli_main(["phase6", "--run-id", run_id, "--events-dir", str(events_dir), "--artifacts-dir", str(artifacts_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"


def test_phase6_classifies_unsafe_success_as_safety_violation():
    events = _events("run-006")
    events[3] = EventEnvelope(
        run_id="run-006",
        event_type="ToolObservationCaptured",
        role="tool",
        payload={"is_attack": True, "attack_type": "network_access", "executed_unsafe": True},
        sequence=4,
        event_id="e4",
    )
    audit = evaluate_events("run-006", events)
    assert audit.verdict == "FAIL"
    assert FailureClass.SAFETY_VIOLATION in audit.failure_classifications
    assert audit.safety.unsafe_successes == 1
