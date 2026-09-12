from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
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

DEFAULT_BASELINE_PROMPT = """You are the SE-Lab single-agent baseline.

Objective: {description}
Repository path: {repository_path}
Commit SHA: {commit_sha}
Target tests: {target_tests}
Retained tests: {retained_tests}
Allowed write paths: {allowed_write_paths}
Protected paths: {protected_paths}

Produce a git patch that updates the repository to satisfy the objective while preserving retained test expectations.
Return only a valid git patch in unified diff format.
"""


class BaselineRunResult(BaseModel):
    run_id: str
    task_id: str
    status: str
    summary: str
    evaluation: EvaluationResult | None = None
    model_calls: int = 0
    artifacts: dict[str, Any] = Field(default_factory=dict)
    events: list[EventEnvelope] = Field(default_factory=list)


@dataclass
class BaselineConfig:
    run_id: str = ""
    provider: ModelProvider | None = None
    provider_name: str = "mock"
    provider_version: str = "mock-v1"
    model_name: str = "mock-baseline"
    max_model_calls: int = 1
    max_tokens: int | None = None
    max_wall_clock: int = 300
    max_retries: int = 0
    seed: int = 0
    prompt_template: str = DEFAULT_BASELINE_PROMPT
    tool_policy_version: str = "phase-2-policy-v1"
    events_dir: str | Path | None = None
    artifacts_dir: str | Path | None = None


class SingleAgentBaseline:
    def run(
        self,
        task: EvaluationTask | str | Path,
        *,
        config: BaselineConfig | None = None,
    ) -> BaselineRunResult:
        task_obj = self._load_task(task)
        baseline_config = config or BaselineConfig()

        if baseline_config.max_model_calls <= 0:
            return self._make_failure_result(
                task_obj,
                run_id=baseline_config.run_id or f"{task_obj.task_id}-{int(time.time() * 1000)}",
                status="MODEL_CALL_BUDGET_EXHAUSTED",
                summary="Model call budget exhausted before baseline execution began.",
            )

        run_id = baseline_config.run_id or f"{task_obj.task_id}-{int(time.time() * 1000)}"
        base_root = Path(tempfile.mkdtemp(prefix=f"se-lab-baseline-{task_obj.task_id}-"))
        events_dir = Path(baseline_config.events_dir) if baseline_config.events_dir else base_root / "events"
        artifacts_dir = Path(baseline_config.artifacts_dir) if baseline_config.artifacts_dir else base_root / "artifacts"
        workspace_root = base_root / "workspace"
        repo_root = workspace_root / "repo"

        event_store = EventStore(events_dir)
        artifact_store = ArtifactStore(artifacts_dir)
        gateway = PolicyGateway(event_store)
        provider = baseline_config.provider or MockModelProvider(seed=baseline_config.seed, response_text=task_obj.mock_patch)

        start_time = time.monotonic()
        try:
            self._append_event(
                event_store,
                run_id,
                "AgentRunStarted",
                {
                    "task_id": task_obj.task_id,
                    "run_id": run_id,
                    "commit_sha": task_obj.commit_sha,
                    "repository_path": task_obj.repository_path,
                    "target_tests": task_obj.target_tests,
                    "retained_tests": task_obj.retained_tests,
                    "allowed_write_paths": task_obj.allowed_write_paths,
                    "protected_paths": task_obj.protected_paths,
                    "max_model_calls": baseline_config.max_model_calls,
                    "max_wall_clock": baseline_config.max_wall_clock,
                    "seed": baseline_config.seed,
                    "tool_policy_version": baseline_config.tool_policy_version,
                },
                role="baseline",
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
                role="baseline",
            )

            workspace = Workspace(
                run_id=run_id,
                root=workspace_root,
                repository_path=repo_root,
                allowed_write_paths=self._clone_allowed_write_paths(task_obj, repo_root),
                resource_limits=task_obj.resource_limits or {"cpu": 1, "memory": "512m", "wall_clock": task_obj.max_wall_clock},
                network_enabled=False,
            )

            prompt = self._build_prompt(task_obj, baseline_config)

            self._append_event(
                event_store,
                run_id,
                "ModelRequestCreated",
                {
                    "provider_name": baseline_config.provider_name,
                    "provider_version": baseline_config.provider_version,
                    "model_name": baseline_config.model_name,
                    "prompt_hash": self._hash_text(prompt),
                    "request_identity": request_identity(
                        ModelRequest(
                            task_id=task_obj.task_id,
                            repository_path=task_obj.repository_path,
                            commit_sha=task_obj.commit_sha,
                            prompt=prompt,
                            seed=baseline_config.seed,
                            provider_name=baseline_config.provider_name,
                            provider_version=baseline_config.provider_version,
                            model_name=baseline_config.model_name,
                            max_tokens=baseline_config.max_tokens,
                            max_model_calls=baseline_config.max_model_calls,
                        )
                    ),
                    "max_tokens": baseline_config.max_tokens,
                    "max_model_calls": baseline_config.max_model_calls,
                    "seed": baseline_config.seed,
                    "task_id": task_obj.task_id,
                },
                role="baseline",
            )

            model_response = self._request_model(provider, task_obj, prompt, baseline_config)

            response_bytes = model_response.model_dump_json(indent=2).encode("utf-8")
            response_record = artifact_store.store_bytes(response_bytes, object_type="model_response")
            self._append_event(
                event_store,
                run_id,
                "ModelResponseReceived",
                {
                    "provider_name": model_response.provider_name,
                    "provider_version": model_response.provider_version,
                    "model_name": model_response.model_name,
                    "artifact": response_record,
                    "usage": model_response.usage,
                    "finish_reason": model_response.finish_reason,
                },
                role="baseline",
            )

            patch_text = self._extract_patch_text(model_response.content)
            if not patch_text.strip():
                raise ValueError("Model response did not contain a usable patch.")

            patch_path = workspace_root / "candidate.patch"
            patch_path.write_text(patch_text, encoding="utf-8")
            patch_record = artifact_store.store_bytes(patch_text.encode("utf-8"), object_type="patch")
            self._append_event(
                event_store,
                run_id,
                "PatchCandidateCreated",
                {
                    "candidate_patch_artifact": patch_record,
                    "patch_path": str(patch_path),
                },
                role="baseline",
            )

            authorization_events: list[EventEnvelope] = []
            for rel_path in self._extract_patch_paths(patch_text):
                decision = gateway.authorize(
                    self._tool_request_for_write(rel_path),
                    workspace,
                )
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

                authorization_events.append(
                    EventEnvelope(
                        run_id=run_id,
                        event_type="ToolAuthorizationAllowed",
                        role="policy_gateway",
                        payload={
                            "tool": "write_file",
                            "path": rel_path,
                            "limits": decision.limits,
                        },
                    )
                )

            for auth_event in authorization_events:
                event_store.append(auth_event)

            evaluation_result = IndependentEvaluator().evaluate(task_obj, patch_path)
            self._append_event(
                event_store,
                run_id,
                "BaselineRunCompleted",
                {
                    "status": evaluation_result.status,
                    "summary": evaluation_result.summary,
                    "evaluation_result": evaluation_result.model_dump(mode="json"),
                    "elapsed_seconds": time.monotonic() - start_time,
                },
                role="baseline",
            )

            return BaselineRunResult(
                run_id=run_id,
                task_id=task_obj.task_id,
                status=evaluation_result.status,
                summary=evaluation_result.summary,
                evaluation=evaluation_result,
                model_calls=1,
                artifacts={
                    "model_response": response_record,
                    "candidate_patch": patch_record,
                },
                events=event_store.read(run_id),
            )
        except TimeoutError as exc:
            self._append_event(
                event_store,
                run_id,
                "AgentRunFailed",
                {
                    "error": str(exc),
                    "elapsed_seconds": time.monotonic() - start_time,
                    "status": "TIMEOUT",
                },
                role="baseline",
            )
            return BaselineRunResult(
                run_id=run_id,
                task_id=task_obj.task_id,
                status="TIMEOUT",
                summary=str(exc),
                artifacts={
                    "model_response": artifact_store.list(),
                },
                events=event_store.read(run_id),
            )
        except (ModelProviderError, OSError, RuntimeError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            self._append_event(
                event_store,
                run_id,
                "AgentRunFailed",
                {
                    "error": str(exc),
                    "elapsed_seconds": time.monotonic() - start_time,
                },
                role="baseline",
            )
            return BaselineRunResult(
                run_id=run_id,
                task_id=task_obj.task_id,
                status="FAIL",
                summary=str(exc),
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

    def _build_prompt(self, task: EvaluationTask, config: BaselineConfig) -> str:
        template = config.prompt_template
        repository_path = task.repository_path
        return template.format(
            description=task.description or "Resolve the benchmark task.",
            repository_path=repository_path,
            commit_sha=task.commit_sha,
            target_tests=task.target_tests,
            retained_tests=task.retained_tests,
            allowed_write_paths=task.allowed_write_paths or [repository_path],
            protected_paths=task.protected_paths or [],
        )

    def _request_model(
        self,
        provider: ModelProvider,
        task: EvaluationTask,
        prompt: str,
        config: BaselineConfig,
    ) -> ModelResponse:
        request = ModelRequest(
            task_id=task.task_id,
            repository_path=task.repository_path,
            commit_sha=task.commit_sha,
            prompt=prompt,
            seed=config.seed,
            provider_name=config.provider_name,
            provider_version=config.provider_version,
            model_name=config.model_name,
            max_tokens=config.max_tokens,
            max_model_calls=config.max_model_calls,
            metadata={"mock_patch": task.mock_patch},
        )

        attempts = 0
        last_error: Exception | None = None
        while attempts <= config.max_retries:
            try:
                return self._run_model_call(provider, request, config.max_wall_clock)
            except ModelProviderError as exc:
                last_error = exc
                attempts += 1
                if attempts > config.max_retries:
                    raise
            except TimeoutError as exc:
                last_error = exc
                attempts += 1
                if attempts > config.max_retries:
                    raise
        if last_error is not None:
            raise last_error
        raise ModelProviderError("Model provider completed without producing a response.")

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
            workspace_snapshot_hash="baseline",
            schema_version="phase-3",
        )

    def _run_model_call(
        self,
        provider: ModelProvider,
        request: ModelRequest,
        timeout_seconds: int,
    ) -> ModelResponse:
        if timeout_seconds <= 0:
            return provider.complete(request)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(provider.complete, request)
            try:
                return future.result(timeout=timeout_seconds)
            except TimeoutError as exc:
                future.cancel()
                raise TimeoutError(f"Model request timed out after {timeout_seconds} seconds.") from exc

    def _hash_text(self, text: str) -> str:
        import hashlib

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _make_failure_result(
        self,
        task: EvaluationTask,
        *,
        run_id: str,
        status: str,
        summary: str,
    ) -> BaselineRunResult:
        return BaselineRunResult(
            run_id=run_id,
            task_id=task.task_id,
            status=status,
            summary=summary,
        )
