from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from se_lab.artifacts.store import ArtifactStore
from se_lab.contracts import EvaluationResult, EventEnvelope, ToolRequest
from se_lab.evaluation import EvaluationTask, IndependentEvaluator
from se_lab.events.store import EventStore
from se_lab.models.provider import (
    MockModelProvider,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    request_identity,
)
from se_lab.policy.gateway import PolicyGateway
from se_lab.runtime.workspace import Workspace


class PlannerOutput(BaseModel):
    run_id: str
    task_id: str
    summary: str
    plan_steps: list[str] = Field(default_factory=list)
    affected_paths: list[str] = Field(default_factory=list)
    expected_tests: list[str] = Field(default_factory=list)
    retained_tests: list[str] = Field(default_factory=list)
    artifact_references: list[str] = Field(default_factory=list)
    schema_version: str = "phase-4"


class ImplementerOutput(BaseModel):
    run_id: str
    task_id: str
    summary: str
    patch_artifact: str | None = None
    affected_paths: list[str] = Field(default_factory=list)
    artifact_references: list[str] = Field(default_factory=list)
    schema_version: str = "phase-4"


class TesterOutput(BaseModel):
    run_id: str
    task_id: str
    summary: str
    status: str
    target_run: dict[str, Any] = Field(default_factory=dict)
    retained_run: dict[str, Any] = Field(default_factory=dict)
    artifact_references: list[str] = Field(default_factory=list)
    schema_version: str = "phase-4"


class ReviewerOutput(BaseModel):
    run_id: str
    task_id: str
    decision: str
    rationale: str
    issues: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)
    evidence_artifacts: list[str] = Field(default_factory=list)
    schema_version: str = "phase-4"


class PlannerToImplementerHandoff(BaseModel):
    run_id: str
    source_role: str = "planner"
    destination_role: str = "implementer"
    payload: PlannerOutput
    artifact_references: list[str] = Field(default_factory=list)
    schema_version: str = "phase-4"


class ImplementerToTesterHandoff(BaseModel):
    run_id: str
    source_role: str = "implementer"
    destination_role: str = "tester"
    payload: ImplementerOutput
    artifact_references: list[str] = Field(default_factory=list)
    schema_version: str = "phase-4"


class TesterToReviewerHandoff(BaseModel):
    run_id: str
    source_role: str = "tester"
    destination_role: str = "reviewer"
    payload: TesterOutput
    artifact_references: list[str] = Field(default_factory=list)
    schema_version: str = "phase-4"


class ReviewerToImplementerHandoff(BaseModel):
    run_id: str
    source_role: str = "reviewer"
    destination_role: str = "implementer"
    payload: ReviewerOutput
    artifact_references: list[str] = Field(default_factory=list)
    schema_version: str = "phase-4"


@dataclass
class MultiAgentConfig:
    run_id: str = ""
    provider: ModelProvider | None = None
    provider_name: str = "mock"
    provider_version: str = "mock-v1"
    model_name: str = "mock-baseline"
    max_model_calls: int = 4
    max_tokens: int | None = None
    max_tool_calls: int = 10
    max_retries: int = 1
    max_wall_clock: int = 300
    seed: int = 0
    events_dir: str | Path | None = None
    artifacts_dir: str | Path | None = None


class MultiAgentRunResult(BaseModel):
    run_id: str
    task_id: str
    status: str
    summary: str
    evaluation: EvaluationResult | None = None
    role_summaries: dict[str, str] = Field(default_factory=dict)
    budgets: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    events: list[EventEnvelope] = Field(default_factory=list)


class MultiAgentWorkflow:
    def run(
        self,
        task: EvaluationTask | str | Path,
        *,
        config: MultiAgentConfig | None = None,
    ) -> MultiAgentRunResult:
        task_obj = self._load_task(task)
        workflow_config = config or MultiAgentConfig()

        if workflow_config.max_model_calls <= 0:
            return self._make_failure_result(
                task_obj,
                run_id=workflow_config.run_id or f"{task_obj.task_id}-{int(time.time() * 1000)}",
                status="MODEL_CALL_BUDGET_EXHAUSTED",
                summary="Model call budget exhausted before multi-agent execution began.",
            )

        run_id = workflow_config.run_id or f"{task_obj.task_id}-{int(time.time() * 1000)}"
        base_root = Path(tempfile.mkdtemp(prefix=f"se-lab-multi-agent-{task_obj.task_id}-"))
        events_dir = Path(workflow_config.events_dir) if workflow_config.events_dir else base_root / "events"
        artifacts_dir = Path(workflow_config.artifacts_dir) if workflow_config.artifacts_dir else base_root / "artifacts"
        workspace_root = base_root / "workspace"
        repo_root = workspace_root / "repo"

        event_store = EventStore(events_dir)
        artifact_store = ArtifactStore(artifacts_dir)
        gateway = PolicyGateway(event_store)
        provider = workflow_config.provider or MockModelProvider(seed=workflow_config.seed)

        model_calls_used = 0
        tool_calls_used = 0
        retries_used = 0
        role_summaries: dict[str, str] = {}
        start_time = time.monotonic()

        try:
            self._append_event(
                event_store,
                run_id,
                "AgentStarted",
                {
                    "role": "planner",
                    "task_id": task_obj.task_id,
                    "run_id": run_id,
                    "commit_sha": task_obj.commit_sha,
                    "workflow": "phase-4-multi-agent",
                },
                role="planner",
            )

            self._materialize_repository(task_obj, repo_root)
            self._append_event(
                event_store,
                run_id,
                "RepositoryMaterialized",
                {
                    "repository_path": str(repo_root),
                    "commit_sha": task_obj.commit_sha,
                },
                role="planner",
            )

            workspace = Workspace(
                run_id=run_id,
                root=workspace_root,
                repository_path=repo_root,
                allowed_write_paths=self._clone_allowed_write_paths(task_obj, repo_root),
                resource_limits=task_obj.resource_limits or {"cpu": 1, "memory": "512m", "wall_clock": task_obj.max_wall_clock},
                network_enabled=False,
            )

            planner_output = self._run_planner(task_obj, provider, workflow_config, event_store, artifact_store, run_id)
            model_calls_used += 1
            role_summaries["planner"] = planner_output.summary
            self._append_event(
                event_store,
                run_id,
                "HandoffCreated",
                {
                    "source_role": "planner",
                    "destination_role": "implementer",
                    "payload": planner_output.model_dump(mode="json"),
                    "artifact_references": planner_output.artifact_references,
                },
                role="planner",
            )

            reviewer_output: ReviewerOutput | None = None
            revision_instructions: list[str] = []
            patch_path: Path | None = None
            patch_artifact_ref: dict[str, Any] | None = None

            while True:
                implementer_output, patch_path, patch_artifact_ref = self._run_implementer(
                    task_obj,
                    planner_output,
                    provider,
                    workflow_config,
                    event_store,
                    artifact_store,
                    run_id,
                    workspace,
                    gateway,
                    tool_calls_used,
                    revision_instructions,
                )
                model_calls_used += 1
                tool_calls_used += len(implementer_output.affected_paths)
                role_summaries["implementer"] = implementer_output.summary

                self._append_event(
                    event_store,
                    run_id,
                    "HandoffCreated",
                    {
                        "source_role": "implementer",
                        "destination_role": "tester",
                        "payload": implementer_output.model_dump(mode="json"),
                        "artifact_references": implementer_output.artifact_references,
                    },
                    role="implementer",
                )

                self._append_event(
                    event_store,
                    run_id,
                    "AgentStarted",
                    {
                        "role": "tester",
                        "task_id": task_obj.task_id,
                        "run_id": run_id,
                    },
                    role="tester",
                )

                tester_output = self._run_tester(
                    task_obj,
                    patch_path,
                    workspace,
                    gateway,
                    event_store,
                    artifact_store,
                    run_id,
                    workflow_config,
                )
                tool_calls_used += 1
                role_summaries["tester"] = tester_output.summary
                self._append_event(
                    event_store,
                    run_id,
                    "HandoffCreated",
                    {
                        "source_role": "tester",
                        "destination_role": "reviewer",
                        "payload": tester_output.model_dump(mode="json"),
                        "artifact_references": tester_output.artifact_references,
                    },
                    role="tester",
                )

                self._append_event(
                    event_store,
                    run_id,
                    "AgentStarted",
                    {
                        "role": "reviewer",
                        "task_id": task_obj.task_id,
                        "run_id": run_id,
                    },
                    role="reviewer",
                )

                reviewer_output = self._run_reviewer(
                    task_obj,
                    tester_output,
                    implementer_output,
                    patch_artifact_ref,
                    event_store,
                    run_id,
                    workflow_config,
                    retries_used,
                )
                role_summaries["reviewer"] = reviewer_output.rationale
                self._append_event(
                    event_store,
                    run_id,
                    "ReviewerDecisionRecorded",
                    reviewer_output.model_dump(mode="json"),
                    role="reviewer",
                )

                if reviewer_output.decision == "ACCEPT":
                    break

                if reviewer_output.decision == "REQUEST_REVISION":
                    if retries_used >= workflow_config.max_retries:
                        failure_summary = (
                            f"Retry budget exhausted after {retries_used} revision cycles. "
                            f"Reviewer requested revision but no further retries remain."
                        )
                        self._append_event(
                            event_store,
                            run_id,
                            "RunTerminated",
                            {
                                "status": "RETRY_EXHAUSTED",
                                "summary": failure_summary,
                                "elapsed_seconds": time.monotonic() - start_time,
                            },
                            role="reviewer",
                        )
                        return MultiAgentRunResult(
                            run_id=run_id,
                            task_id=task_obj.task_id,
                            status="RETRY_EXHAUSTED",
                            summary=failure_summary,
                            budgets={
                                "model_calls_used": model_calls_used,
                                "tool_calls_used": tool_calls_used,
                                "retries_used": retries_used,
                                "elapsed_seconds": round(time.monotonic() - start_time, 3),
                                "max_model_calls": workflow_config.max_model_calls,
                                "max_tool_calls": workflow_config.max_tool_calls,
                                "max_retries": workflow_config.max_retries,
                            },
                            artifacts={
                                "model_response": artifact_store.list(),
                                "candidate_patch": patch_artifact_ref,
                            },
                            events=event_store.read(run_id),
                        )

                    retries_used += 1
                    revision_instructions = reviewer_output.revision_instructions
                    self._append_event(
                        event_store,
                        run_id,
                        "HandoffCreated",
                        {
                            "source_role": "reviewer",
                            "destination_role": "implementer",
                            "payload": reviewer_output.model_dump(mode="json"),
                            "artifact_references": reviewer_output.evidence_artifacts,
                        },
                        role="reviewer",
                    )
                    continue

                if reviewer_output.decision == "REJECT":
                    failure_summary = reviewer_output.rationale
                    self._append_event(
                        event_store,
                        run_id,
                        "RunTerminated",
                        {
                            "status": "FAIL",
                            "summary": failure_summary,
                            "elapsed_seconds": time.monotonic() - start_time,
                        },
                        role="reviewer",
                    )
                    return MultiAgentRunResult(
                        run_id=run_id,
                        task_id=task_obj.task_id,
                        status="FAIL",
                        summary=failure_summary,
                        budgets={
                            "model_calls_used": model_calls_used,
                            "tool_calls_used": tool_calls_used,
                            "retries_used": retries_used,
                            "elapsed_seconds": round(time.monotonic() - start_time, 3),
                            "max_model_calls": workflow_config.max_model_calls,
                            "max_tool_calls": workflow_config.max_tool_calls,
                            "max_retries": workflow_config.max_retries,
                        },
                        artifacts={
                            "model_response": artifact_store.list(),
                            "candidate_patch": patch_artifact_ref,
                        },
                        events=event_store.read(run_id),
                    )

            evaluation_result = IndependentEvaluator().evaluate(task_obj, patch_path)
            self._append_event(
                event_store,
                run_id,
                "EvaluationCompleted",
                {
                    "status": evaluation_result.status,
                    "summary": evaluation_result.summary,
                    "evaluation_result": evaluation_result.model_dump(mode="json"),
                },
                role="evaluator",
            )
            self._append_event(
                event_store,
                run_id,
                "RunTerminated",
                {
                    "status": evaluation_result.status,
                    "summary": evaluation_result.summary,
                    "elapsed_seconds": time.monotonic() - start_time,
                },
                role="evaluator",
            )

            return MultiAgentRunResult(
                run_id=run_id,
                task_id=task_obj.task_id,
                status=evaluation_result.status,
                summary=evaluation_result.summary,
                evaluation=evaluation_result,
                role_summaries=role_summaries,
                budgets={
                    "model_calls_used": model_calls_used,
                    "tool_calls_used": tool_calls_used,
                    "retries_used": retries_used,
                    "elapsed_seconds": round(time.monotonic() - start_time, 3),
                    "max_model_calls": workflow_config.max_model_calls,
                    "max_tool_calls": workflow_config.max_tool_calls,
                    "max_retries": workflow_config.max_retries,
                },
                artifacts={
                    "model_response": artifact_store.list(),
                    "candidate_patch": patch_artifact_ref,
                },
                events=event_store.read(run_id),
            )
        except (ModelProviderError, OSError, RuntimeError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            self._append_event(
                event_store,
                run_id,
                "RunTerminated",
                {
                    "status": "FAIL",
                    "summary": str(exc),
                    "elapsed_seconds": time.monotonic() - start_time,
                },
                role="planner",
            )
            return MultiAgentRunResult(
                run_id=run_id,
                task_id=task_obj.task_id,
                status="FAIL",
                summary=str(exc),
                budgets={
                    "model_calls_used": model_calls_used,
                    "tool_calls_used": tool_calls_used,
                    "retries_used": retries_used,
                    "elapsed_seconds": round(time.monotonic() - start_time, 3),
                    "max_model_calls": workflow_config.max_model_calls,
                    "max_tool_calls": workflow_config.max_tool_calls,
                    "max_retries": workflow_config.max_retries,
                },
                artifacts={
                    "model_response": artifact_store.list(),
                },
                events=event_store.read(run_id),
            )

    def _load_task(self, task: EvaluationTask | str | Path) -> EvaluationTask:
        if isinstance(task, EvaluationTask):
            return task
        task_path = Path(task)
        if not task_path.exists():
            raise FileNotFoundError(f"Task definition not found: {task_path}")
        return EvaluationTask.from_file(task_path)

    def _run_planner(
        self,
        task: EvaluationTask,
        provider: ModelProvider,
        config: MultiAgentConfig,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        run_id: str,
    ) -> PlannerOutput:
        self._append_event(
            event_store,
            run_id,
            "AgentStarted",
            {
                "role": "planner",
                "run_id": run_id,
                "task_id": task.task_id,
            },
            role="planner",
        )

        planner_payload = {
            "run_id": run_id,
            "task_id": task.task_id,
            "summary": f"Plan a bounded implementation for task {task.task_id}.",
            "plan_steps": [
                "inspect repository state and task constraints",
                "prepare a minimal candidate patch",
                "validate the patch through approved tests",
                "request reviewer decision",
            ],
            "affected_paths": task.allowed_write_paths or [task.repository_path],
            "expected_tests": task.target_tests,
            "retained_tests": task.retained_tests,
            "artifact_references": [],
        }

        planner_response = self._request_model(
            provider,
            task,
            run_id,
            event_store,
            artifact_store,
            role="planner",
            metadata={
                "mock_patch": json.dumps(planner_payload),
            },
            config=config,
        )

        try:
            payload = json.loads(planner_response.content)
        except json.JSONDecodeError as exc:
            raise ValueError("Planner model response was not valid JSON.") from exc

        planner_output = PlannerOutput(
            run_id=run_id,
            task_id=task.task_id,
            summary=payload.get("summary", "Planner produced a bounded implementation plan."),
            plan_steps=payload.get("plan_steps", []),
            affected_paths=payload.get("affected_paths", []),
            expected_tests=payload.get("expected_tests", task.target_tests),
            retained_tests=payload.get("retained_tests", task.retained_tests),
            artifact_references=payload.get("artifact_references", []),
        )
        return planner_output

    def _run_implementer(
        self,
        task: EvaluationTask,
        planner_output: PlannerOutput,
        provider: ModelProvider,
        config: MultiAgentConfig,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        run_id: str,
        workspace: Workspace,
        gateway: PolicyGateway,
        tool_calls_used: int,
        revision_instructions: list[str],
    ) -> tuple[ImplementerOutput, Path, dict[str, Any] | None]:
        self._append_event(
            event_store,
            run_id,
            "AgentStarted",
            {
                "role": "implementer",
                "run_id": run_id,
                "task_id": task.task_id,
                "revision_instructions": revision_instructions,
            },
            role="implementer",
        )

        implementer_metadata: dict[str, Any] = {
            "mock_patch": task.mock_patch,
            "planner_summary": planner_output.summary,
            "plan_steps": planner_output.plan_steps,
            "revision_instructions": revision_instructions,
        }
        implementer_response = self._request_model(
            provider,
            task,
            run_id,
            event_store,
            artifact_store,
            role="implementer",
            metadata=implementer_metadata,
            config=config,
        )

        patch_text = self._extract_patch_text(implementer_response.content)
        if not patch_text.strip():
            raise ValueError("Implementer model response did not contain a usable patch.")

        patch_path = workspace.root / "candidate.patch"
        patch_path.write_text(patch_text, encoding="utf-8")
        patch_artifact = artifact_store.store_bytes(patch_text.encode("utf-8"), object_type="patch")
        self._append_event(
            event_store,
            run_id,
            "PatchCandidateCreated",
            {
                "candidate_patch_artifact": patch_artifact,
                "patch_path": str(patch_path),
            },
            role="implementer",
        )

        self._append_event(
            event_store,
            run_id,
            "ToolStarted",
            {
                "role": "implementer",
                "tool_name": "write_file",
                "path": "candidate.patch",
            },
            role="implementer",
        )

        for rel_path in self._extract_patch_paths(patch_text):
            if tool_calls_used >= config.max_tool_calls:
                raise RuntimeError("Tool call budget exhausted before implementing the candidate patch.")
            tool_calls_used += 1
            decision = gateway.authorize(self._tool_request_for_write(rel_path), workspace)
            if decision.decision != "allow":
                self._append_event(
                    event_store,
                    run_id,
                    "ToolAuthorizationDenied",
                    {
                        "tool": "write_file",
                        "path": rel_path,
                        "reason": decision.reason,
                        "limits": decision.limits,
                    },
                    role="policy_gateway",
                )
                raise RuntimeError(f"Patch rejected by policy: {decision.reason}")

            self._append_event(
                event_store,
                run_id,
                "ToolObservationCaptured",
                {
                    "tool": "write_file",
                    "path": rel_path,
                    "decision": decision.model_dump(mode="json"),
                },
                role="implementer",
            )

        implementer_output = ImplementerOutput(
            run_id=run_id,
            task_id=task.task_id,
            summary=f"Prepared candidate patch for task {task.task_id}.",
            patch_artifact=patch_artifact["sha256"],
            affected_paths=self._extract_patch_paths(patch_text),
            artifact_references=[patch_artifact["sha256"]],
        )
        return implementer_output, patch_path, patch_artifact

    def _run_tester(
        self,
        task: EvaluationTask,
        patch_path: Path,
        workspace: Workspace,
        gateway: PolicyGateway,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        run_id: str,
        config: MultiAgentConfig,
    ) -> TesterOutput:
        self._append_event(
            event_store,
            run_id,
            "PatchApplied",
            {
                "patch_path": str(patch_path),
                "repository_path": str(workspace.repository_path),
            },
            role="tester",
        )
        self._apply_patch(workspace.repository_path, patch_path)

        self._append_event(
            event_store,
            run_id,
            "TestRunStarted",
            {
                "suite": "target",
                "tests": task.target_tests,
            },
            role="tester",
        )

        decision = gateway.authorize(
            ToolRequest(
                tool_name="run_tests",
                arguments={"command": ["pytest", "-q", *task.target_tests]},
                workspace_snapshot_hash="phase-4",
                schema_version="phase-4",
            ),
            workspace,
        )
        if decision.decision != "allow":
            raise RuntimeError(f"Tester could not run approved tests: {decision.reason}")

        self._append_event(
            event_store,
            run_id,
            "ToolStarted",
            {
                "role": "tester",
                "tool_name": "run_tests",
                "command": ["pytest", "-q", *task.target_tests],
            },
            role="tester",
        )

        target_run = self._run_test_suite(workspace.repository_path, task.target_tests, config.max_wall_clock)
        self._append_event(
            event_store,
            run_id,
            "TestRunCompleted",
            {
                "suite": "target",
                "command": target_run["command"],
                "exit_code": target_run["exit_code"],
                "status": target_run["status"],
                "stdout": target_run["stdout"],
                "stderr": target_run["stderr"],
                "result": target_run["result"],
            },
            role="tester",
        )
        self._append_event(
            event_store,
            run_id,
            "ToolObservationCaptured",
            {
                "tool": "run_tests",
                "suite": "target",
                "result": target_run,
            },
            role="tester",
        )

        retained_run = self._run_test_suite(workspace.repository_path, task.retained_tests, config.max_wall_clock)
        self._append_event(
            event_store,
            run_id,
            "TestRunCompleted",
            {
                "suite": "retained",
                "command": retained_run["command"],
                "exit_code": retained_run["exit_code"],
                "status": retained_run["status"],
                "stdout": retained_run["stdout"],
                "stderr": retained_run["stderr"],
                "result": retained_run["result"],
            },
            role="tester",
        )
        self._append_event(
            event_store,
            run_id,
            "ToolObservationCaptured",
            {
                "tool": "run_tests",
                "suite": "retained",
                "result": retained_run,
            },
            role="tester",
        )

        target_ok = target_run["status"] == "PASS"
        retained_ok = retained_run["status"] == "PASS"
        if target_ok and retained_ok:
            summary = "Target and retained tests passed after the candidate patch was applied."
            status = "PASS"
        elif target_run["status"] == "TIMEOUT" or retained_run["status"] == "TIMEOUT":
            summary = "Test timeout encountered during multi-agent validation."
            status = "TIMEOUT"
        elif target_run["status"] == "INFRASTRUCTURE_FAILURE" or retained_run["status"] == "INFRASTRUCTURE_FAILURE":
            summary = "Infrastructure failure occurred while running the approved test suite."
            status = "INFRASTRUCTURE_FAILURE"
        else:
            summary = "Target or retained tests failed after the candidate patch was applied."
            status = "FAIL"

        tester_output = TesterOutput(
            run_id=run_id,
            task_id=task.task_id,
            summary=summary,
            status=status,
            target_run=target_run,
            retained_run=retained_run,
            artifact_references=[str(patch_path)],
        )
        return tester_output

    def _run_reviewer(
        self,
        task: EvaluationTask,
        tester_output: TesterOutput,
        implementer_output: ImplementerOutput,
        patch_artifact: dict[str, Any] | None,
        event_store: EventStore,
        run_id: str,
        config: MultiAgentConfig,
        retries_used: int,
    ) -> ReviewerOutput:
        issues: list[str] = []
        revision_instructions: list[str] = []
        rationale = "Reviewed the candidate patch and test evidence."

        if tester_output.status == "PASS":
            decision = "ACCEPT"
            rationale = "Candidate patch satisfied the target and retained tests."
        elif tester_output.status in {"TIMEOUT", "INFRASTRUCTURE_FAILURE"}:
            decision = "REJECT"
            rationale = tester_output.summary
            issues.append(tester_output.summary)
        else:
            decision = "REQUEST_REVISION"
            rationale = "Candidate patch failed validation; a bounded revision is requested."
            issues.append(tester_output.summary)
            revision_instructions.append("Rework the candidate patch to satisfy the target and retained tests.")
            revision_instructions.append("Preserve the existing evaluator and policy boundaries.")

        evidence_artifacts = [
            patch_artifact["path"] if patch_artifact else "",
            *implementer_output.artifact_references,
        ]
        evidence_artifacts = [item for item in evidence_artifacts if item]

        reviewer_output = ReviewerOutput(
            run_id=run_id,
            task_id=task.task_id,
            decision=decision,
            rationale=rationale,
            issues=issues,
            revision_instructions=revision_instructions,
            evidence_artifacts=evidence_artifacts,
        )

        self._append_event(
            event_store,
            run_id,
            "ReviewerDecisionRecorded",
            reviewer_output.model_dump(mode="json"),
            role="reviewer",
        )
        return reviewer_output

    def _role_prompt(self, task: EvaluationTask, role: str, metadata: dict[str, Any]) -> str:
        output = {
            "planner": "Return a JSON implementation plan only.",
            "implementer": "Return a valid unified git patch only.",
            "tester": "Return a JSON test-result summary only.",
            "reviewer": "Return a JSON review decision only.",
        }.get(role, "Return a concise structured response only.")
        return (
            f"You are the SE-Lab {role} agent.\n"
            f"Task: {task.description}\n"
            f"Task ID: {task.task_id}\n"
            f"Repository commit: {task.commit_sha}\n"
            f"Target tests: {task.target_tests}\n"
            f"Retained tests: {task.retained_tests}\n"
            f"Allowed write paths: {task.allowed_write_paths}\n"
            f"Protected paths: {task.protected_paths}\n"
            f"Context: {metadata.get('planner_summary', '')}\n"
            f"{output}"
        )

    def _request_model(
        self,
        provider: ModelProvider,
        task: EvaluationTask,
        run_id: str,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        *,
        role: str,
        metadata: dict[str, Any],
        config: MultiAgentConfig,
    ) -> ModelResponse:
        request = ModelRequest(
            task_id=task.task_id,
            repository_path=task.repository_path,
            commit_sha=task.commit_sha,
            prompt=self._role_prompt(task, role, metadata),
            seed=config.seed,
            provider_name=config.provider_name,
            provider_version=config.provider_version,
            model_name=config.model_name,
            max_tokens=config.max_tokens,
            max_model_calls=config.max_model_calls,
            metadata={**metadata, "role": role},
        )

        self._append_event(
            event_store,
            run_id,
            "ModelRequestIssued",
            {
                "role": role,
                "provider_name": config.provider_name,
                "provider_version": config.provider_version,
                "model_name": config.model_name,
                "request_identity": request_identity(request),
                "metadata": metadata,
            },
            role=role,
        )

        response = provider.complete(request)

        response_bytes = response.model_dump_json(indent=2).encode("utf-8")
        response_record = artifact_store.store_bytes(response_bytes, object_type=f"model_response_{role}")
        self._append_event(
            event_store,
            run_id,
            "ModelResponseReceived",
            {
                "role": role,
                "provider_name": response.provider_name,
                "provider_version": response.provider_version,
                "model_name": response.model_name,
                "artifact": response_record,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
            },
            role=role,
        )

        return response

    def _append_event(
        self,
        event_store: EventStore,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        *,
        role: str,
    ) -> None:
        event_store.append(
            EventEnvelope(
                run_id=run_id,
                event_type=event_type,
                role=role,
                payload=payload,
            )
        )

    def _materialize_repository(self, task: EvaluationTask, repo_root: Path) -> None:
        source_root = Path(task.repository_path)
        if not source_root.exists():
            raise FileNotFoundError(f"Repository fixture not found at {source_root}")

        git_executable = shutil.which("git")
        if git_executable is None:
            raise RuntimeError("git is required to materialize evaluation repositories")

        clone = subprocess.run(
            [git_executable, "clone", "--quiet", str(source_root), str(repo_root)],
            capture_output=True,
            text=True,
            check=False,
        )
        if clone.returncode != 0:
            raise RuntimeError(f"Failed to clone repository fixture: {clone.stderr}")

        if task.commit_sha:
            checkout = subprocess.run(
                [git_executable, "-C", str(repo_root), "checkout", "--quiet", task.commit_sha],
                capture_output=True,
                text=True,
                check=False,
            )
            if checkout.returncode != 0:
                raise RuntimeError(f"Failed to checkout commit {task.commit_sha}: {checkout.stderr}")

    def _apply_patch(self, repo_root: Path, patch_path: Path) -> None:
        git_executable = shutil.which("git")
        if git_executable is None:
            raise RuntimeError("git is required to apply candidate patches")

        apply = subprocess.run(
            [git_executable, "-C", str(repo_root), "apply", str(patch_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if apply.returncode != 0:
            raise RuntimeError(f"Failed to apply candidate patch: {apply.stderr}")

    def _clone_allowed_write_paths(self, task: EvaluationTask, repo_root: Path) -> list[Path]:
        source_root = Path(task.repository_path).resolve()
        allowed_paths: list[Path] = []

        for allowed in task.allowed_write_paths or [repo_root]:
            candidate = Path(allowed)
            if not candidate.is_absolute():
                candidate = (source_root / candidate).resolve(strict=False)
            else:
                candidate = candidate.resolve(strict=False)

            if candidate == source_root:
                allowed_paths.append(repo_root)
                continue

            if candidate.is_relative_to(source_root):
                relative = candidate.relative_to(source_root)
                allowed_paths.append((repo_root / relative).resolve())
                continue

            allowed_paths.append(candidate)

        return allowed_paths

    def _extract_patch_text(self, content: str) -> str:
        stripped = content.strip()
        if not stripped:
            raise ValueError("Model response did not include a patch.")
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError("Malformed model response: expected patch content or JSON object containing a patch.") from exc
            if isinstance(payload, dict):
                for key in ("patch", "candidate_patch"):
                    candidate = payload.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", candidate)
            raise ValueError("Malformed model response: JSON payload did not contain a patch string.")
        return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", content)

    def _extract_patch_paths(self, patch_text: str) -> list[str]:
        paths: list[str] = []
        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                candidate = line[4:]
                if candidate in {"/dev/null", "---"}:
                    continue
                candidate = candidate.removeprefix("b/")
                candidate = candidate.removeprefix("a/")
                if candidate.startswith("/"):
                    raise ValueError(f"Patch attempts to write outside the workspace: {candidate}")
                paths.append(candidate)

        normalized: list[str] = []
        for raw_path in paths:
            rel_path = raw_path.strip().replace("\\", "/")
            candidate = Path(rel_path)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"Patch attempts to escape the workspace: {rel_path}")
            normalized.append(rel_path)
        return normalized

    def _tool_request_for_write(self, rel_path: str) -> ToolRequest:
        return ToolRequest(
            tool_name="write_file",
            arguments={"path": rel_path},
            workspace_snapshot_hash="phase-4",
            schema_version="phase-4",
        )

    def _run_test_suite(self, repo_root: Path, tests: list[str], timeout_seconds: int) -> dict[str, Any]:
        if not tests:
            return {
                "command": [],
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "duration_seconds": 0.0,
                "status": "PASS",
                "result": {"passed": 0, "failed": 0, "skipped": 0, "errors": 0},
            }

        command = [shutil.which("python") or "python", "-m", "pytest", "-q", *tests]
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            duration = time.monotonic() - start
            result = self._parse_pytest_output(completed.stdout, completed.stderr)
            return {
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "duration_seconds": duration,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "result": result,
            }
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            return {
                "command": command,
                "exit_code": None,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "duration_seconds": duration,
                "status": "TIMEOUT",
                "result": {"passed": 0, "failed": 0, "skipped": 0, "errors": 1},
            }
        except OSError as exc:
            duration = time.monotonic() - start
            return {
                "command": command,
                "exit_code": None,
                "stdout": "",
                "stderr": str(exc),
                "duration_seconds": duration,
                "status": "INFRASTRUCTURE_FAILURE",
                "error": str(exc),
                "result": {"passed": 0, "failed": 0, "skipped": 0, "errors": 1},
            }

    def _parse_pytest_output(self, stdout: str, stderr: str) -> dict[str, int]:
        combined = (stdout or "") + "\n" + (stderr or "")
        result = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        for label in ("passed", "failed", "skipped", "error", "errors"):
            match = __import__("re").search(rf"(\d+)\s+{label}\b", combined)
            if match:
                if label in {"error", "errors"}:
                    result["errors"] = int(match.group(1))
                else:
                    result[label] = int(match.group(1))
        if result == {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}:
            if "passed" in combined:
                match = __import__("re").search(r"(\d+) passed", combined)
                if match:
                    result["passed"] = int(match.group(1))
            if "failed" in combined:
                match = __import__("re").search(r"(\d+) failed", combined)
                if match:
                    result["failed"] = int(match.group(1))
            if "skipped" in combined:
                match = __import__("re").search(r"(\d+) skipped", combined)
                if match:
                    result["skipped"] = int(match.group(1))
        return result

    def _make_failure_result(
        self,
        task: EvaluationTask,
        *,
        run_id: str,
        status: str,
        summary: str,
    ) -> MultiAgentRunResult:
        return MultiAgentRunResult(
            run_id=run_id,
            task_id=task.task_id,
            status=status,
            summary=summary,
        )
