from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from se_lab.artifacts.store import ArtifactStore
from se_lab.contracts import EvaluationResult, EventEnvelope, ToolRequest
from se_lab.events.store import EventStore
from se_lab.policy.gateway import PolicyGateway
from se_lab.runtime.workspace import Workspace

DEFAULT_PROTECTED_PATHS = [
    "evaluation",
    "src/se_lab/evaluation",
    "src/se_lab/agents",
    "docker/worker.Dockerfile",
    "docker/seccomp.json",
]


@dataclass
class EvaluationTask:
    task_id: str
    repository_path: str
    commit_sha: str
    target_tests: list[str] = field(default_factory=list)
    retained_tests: list[str] = field(default_factory=list)
    protected_paths: list[str] = field(default_factory=list)
    max_wall_clock: int = 300
    resource_limits: dict[str, Any] = field(default_factory=dict)
    allowed_write_paths: list[str] = field(default_factory=list)
    description: str = ""
    mock_patch: str = ""
    difficulty: str = "smoke"
    category: str = "curated"
    expected_behavior: str = ""
    adversarial: bool = False

    @classmethod
    def from_file(cls, path: str | Path) -> EvaluationTask:
        task_path = Path(path)
        data = json.loads(task_path.read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IndependentEvaluator:
    def evaluate(self, task: EvaluationTask | str | Path, patch: str | Path) -> EvaluationResult:
        task_obj = self._load_task(task)
        patch_path = Path(patch)

        run_id = f"{task_obj.task_id}-{int(time.time() * 1000)}"
        work_root = Path(tempfile.mkdtemp(prefix=f"se-lab-eval-{task_obj.task_id}-"))
        repo_root = work_root / "repo"
        events_dir = work_root / "events"
        artifacts_dir = work_root / "artifacts"

        event_store = EventStore(events_dir)
        artifact_store = ArtifactStore(artifacts_dir)
        gateway = PolicyGateway(event_store)

        try:
            self._store_event(
                event_store,
                run_id,
                "TestRunStarted",
                {
                    "task_id": task_obj.task_id,
                    "commit_sha": task_obj.commit_sha,
                    "target_tests": task_obj.target_tests,
                    "retained_tests": task_obj.retained_tests,
                    "description": task_obj.description,
                },
                role="evaluator",
            )

            self._materialize_repository(task_obj, repo_root)

            workspace = Workspace(
                run_id=run_id,
                root=work_root,
                repository_path=repo_root,
                allowed_write_paths=self._clone_allowed_write_paths(task_obj, repo_root),
                resource_limits=task_obj.resource_limits or {"cpu": 1, "memory": "512m", "wall_clock": task_obj.max_wall_clock},
                network_enabled=False,
            )

            protected_prefixes = {str(Path(item).as_posix()) for item in (task_obj.protected_paths or []) + DEFAULT_PROTECTED_PATHS}

            patch_bytes = patch_path.read_bytes()
            self._store_artifact(event_store, artifact_store, run_id, "candidate_patch", patch_bytes, "patch")

            if not patch_bytes.strip():
                return self._finalize_failure(
                    event_store,
                    artifact_store,
                    run_id,
                    task_obj,
                    "FAIL",
                    "Malformed patch: empty patch input.",
                    target_tests=task_obj.target_tests,
                    retained_tests=task_obj.retained_tests,
                )

            normalized_paths = self._extract_patch_paths(patch_path.read_text(encoding="utf-8"))
            if not normalized_paths:
                return self._finalize_failure(
                    event_store,
                    artifact_store,
                    run_id,
                    task_obj,
                    "FAIL",
                    "Malformed patch: no file paths found in patch.",
                    target_tests=task_obj.target_tests,
                    retained_tests=task_obj.retained_tests,
                )

            for rel_path in normalized_paths:
                if self._is_protected_path(rel_path, protected_prefixes):
                    return self._finalize_failure(
                        event_store,
                        artifact_store,
                        run_id,
                        task_obj,
                        "FAIL",
                        f"Patch attempts to modify protected evaluator asset: {rel_path}",
                        target_tests=task_obj.target_tests,
                        retained_tests=task_obj.retained_tests,
                    )

                try:
                    decision = gateway.authorize(
                        self._tool_request_for_write(rel_path),
                        workspace,
                    )
                except (RuntimeError, ValueError, TypeError, OSError) as exc:
                    return self._finalize_failure(
                        event_store,
                        artifact_store,
                        run_id,
                        task_obj,
                        "FAIL",
                        f"Malformed patch: unable to validate patch path {rel_path}: {exc}",
                        target_tests=task_obj.target_tests,
                        retained_tests=task_obj.retained_tests,
                    )

                if decision.decision != "allow":
                    return self._finalize_failure(
                        event_store,
                        artifact_store,
                        run_id,
                        task_obj,
                        "FAIL",
                        f"Patch rejected by policy: {decision.reason}",
                        target_tests=task_obj.target_tests,
                        retained_tests=task_obj.retained_tests,
                    )

            self._apply_patch(repo_root, patch_path)
            diff_bytes = self._collect_diff(repo_root)
            self._store_artifact(event_store, artifact_store, run_id, "patch_diff", diff_bytes, "diff")

            target_run = self._run_test_suite(repo_root, task_obj.target_tests, task_obj.max_wall_clock)
            self._store_event(
                event_store,
                run_id,
                "TestRunCompleted",
                {
                    "suite": "target",
                    "command": target_run["command"],
                    "exit_code": target_run["exit_code"],
                    "duration_seconds": target_run["duration_seconds"],
                    "stdout": target_run["stdout"],
                    "stderr": target_run["stderr"],
                    "result": target_run["result"],
                },
                role="evaluator",
            )

            if target_run["status"] == "TIMEOUT":
                return self._finalize_failure(
                    event_store,
                    artifact_store,
                    run_id,
                    task_obj,
                    "TIMEOUT",
                    f"Target test run timed out after {task_obj.max_wall_clock} seconds.",
                    target_tests=task_obj.target_tests,
                    retained_tests=task_obj.retained_tests,
                    target_run=target_run,
                )

            if target_run["status"] == "INFRASTRUCTURE_FAILURE":
                return self._finalize_failure(
                    event_store,
                    artifact_store,
                    run_id,
                    task_obj,
                    "INFRASTRUCTURE_FAILURE",
                    target_run["error"],
                    target_tests=task_obj.target_tests,
                    retained_tests=task_obj.retained_tests,
                    target_run=target_run,
                )

            if target_run["exit_code"] != 0:
                return self._finalize_failure(
                    event_store,
                    artifact_store,
                    run_id,
                    task_obj,
                    "FAIL",
                    "Target tests failed after applying the patch.",
                    target_tests=task_obj.target_tests,
                    retained_tests=task_obj.retained_tests,
                    target_run=target_run,
                )

            retained_run = self._run_test_suite(repo_root, task_obj.retained_tests, task_obj.max_wall_clock)
            self._store_event(
                event_store,
                run_id,
                "TestRunCompleted",
                {
                    "suite": "retained",
                    "command": retained_run["command"],
                    "exit_code": retained_run["exit_code"],
                    "duration_seconds": retained_run["duration_seconds"],
                    "stdout": retained_run["stdout"],
                    "stderr": retained_run["stderr"],
                    "result": retained_run["result"],
                },
                role="evaluator",
            )

            if retained_run["status"] == "TIMEOUT":
                return self._finalize_failure(
                    event_store,
                    artifact_store,
                    run_id,
                    task_obj,
                    "TIMEOUT",
                    f"Retained test run timed out after {task_obj.max_wall_clock} seconds.",
                    target_tests=task_obj.target_tests,
                    retained_tests=task_obj.retained_tests,
                    target_run=target_run,
                    retained_run=retained_run,
                )

            if retained_run["status"] == "INFRASTRUCTURE_FAILURE":
                return self._finalize_failure(
                    event_store,
                    artifact_store,
                    run_id,
                    task_obj,
                    "INFRASTRUCTURE_FAILURE",
                    retained_run["error"],
                    target_tests=task_obj.target_tests,
                    retained_tests=task_obj.retained_tests,
                    target_run=target_run,
                    retained_run=retained_run,
                )

            regression_detected = retained_run["exit_code"] != 0
            if regression_detected:
                return self._finalize_failure(
                    event_store,
                    artifact_store,
                    run_id,
                    task_obj,
                    "FAIL",
                    "Retained tests regressed after applying the patch.",
                    target_tests=task_obj.target_tests,
                    retained_tests=task_obj.retained_tests,
                    target_run=target_run,
                    retained_run=retained_run,
                    regression_detected=True,
                )

            return self._finalize_success(
                event_store,
                artifact_store,
                run_id,
                task_obj,
                target_run,
                retained_run,
            )
        except (OSError, RuntimeError, ValueError, TypeError, subprocess.SubprocessError) as exc:
            return self._finalize_failure(
                event_store,
                artifact_store,
                run_id,
                task_obj,
                "INFRASTRUCTURE_FAILURE",
                str(exc),
                target_tests=task_obj.target_tests,
                retained_tests=task_obj.retained_tests,
            )

    def _load_task(self, task: EvaluationTask | str | Path) -> EvaluationTask:
        if isinstance(task, EvaluationTask):
            return task

        task_path = Path(task)
        if not task_path.exists():
            raise FileNotFoundError(f"Task definition not found: {task_path}")

        return EvaluationTask.from_file(task_path)

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

    def _tool_request_for_write(self, rel_path: str) -> ToolRequest:
        request_path = rel_path.replace("\\", "/")
        return ToolRequest(
            tool_name="write_file",
            arguments={"path": request_path},
            workspace_snapshot_hash="evaluator",
            schema_version="phase-3",
        )

    def _extract_patch_paths(self, patch_text: str) -> list[str]:
        # Git may emit terminal color codes when color.ui is configured as
        # always; strip presentation bytes before parsing the unified diff.
        patch_text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", patch_text)
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
            rel_path = raw_path.strip()
            rel_path = rel_path.replace("\\", "/")
            candidate = Path(rel_path)

            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"Patch attempts to escape the workspace: {rel_path}")

            normalized.append(rel_path)
        return normalized

    def _is_protected_path(self, rel_path: str, protected_prefixes: set[str]) -> bool:
        rel_path = rel_path.replace("\\", "/")
        for protected in protected_prefixes:
            protected = protected.replace("\\", "/")
            if rel_path == protected or rel_path.startswith(f"{protected}/"):
                return True
        return False

    def _apply_patch(self, repo_root: Path, patch_path: Path) -> None:
        git_executable = shutil.which("git")
        if git_executable is None:
            raise RuntimeError("git is required to apply evaluation patches")

        patch_text = patch_path.read_text(encoding="utf-8")
        patch_text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", patch_text)
        apply = subprocess.run(
            [git_executable, "-C", str(repo_root), "apply", "-"],
            input=patch_text,
            capture_output=True,
            text=True,
            check=False,
        )
        if apply.returncode != 0:
            raise RuntimeError(f"Failed to apply candidate patch: {apply.stderr}")

    def _collect_diff(self, repo_root: Path) -> bytes:
        git_executable = shutil.which("git")
        if git_executable is None:
            raise RuntimeError("git is required to inspect evaluation diffs")

        diff = subprocess.run(
            [git_executable, "-C", str(repo_root), "diff", "--binary", "--no-ext-diff"],
            capture_output=True,
            text=False,
            check=False,
        )
        if diff.returncode != 0:
            raise RuntimeError(f"Failed to collect evaluation diff: {diff.stderr.decode('utf-8', errors='replace')}")
        return diff.stdout

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

        command = [sys.executable, "-m", "pytest", "-q", *tests]
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
            match = re.search(rf"(\d+)\s+{label}\b", combined)
            if match:
                if label == "error" or label == "errors":
                    result["errors"] = int(match.group(1))
                else:
                    result[label] = int(match.group(1))

        if result == {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}:
            if "passed" in combined:
                match = re.search(r"(\d+) passed", combined)
                if match:
                    result["passed"] = int(match.group(1))
            if "failed" in combined:
                match = re.search(r"(\d+) failed", combined)
                if match:
                    result["failed"] = int(match.group(1))
            if "skipped" in combined:
                match = re.search(r"(\d+) skipped", combined)
                if match:
                    result["skipped"] = int(match.group(1))
        return result

    def _store_event(
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

    def _store_artifact(
        self,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        run_id: str,
        key: str,
        content: bytes,
        object_type: str,
    ) -> None:
        record = artifact_store.store_bytes(content, object_type=object_type)
        self._store_event(
            event_store,
            run_id,
            "ArtifactStored",
            {
                "key": key,
                "artifact": record,
            },
            role="evaluator",
        )

    def _finalize_success(
        self,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        run_id: str,
        task: EvaluationTask,
        target_run: dict[str, Any],
        retained_run: dict[str, Any],
    ) -> EvaluationResult:
        metadata = {
            "task_id": task.task_id,
            "commit_sha": task.commit_sha,
            "status": "PASS",
            "summary": "All target and retained tests passed.",
            "target_tests": task.target_tests,
            "retained_tests": task.retained_tests,
            "artifacts": artifact_store.list(),
        }
        self._store_event(
            event_store,
            run_id,
            "EvaluationCompleted",
            {
                "status": "PASS",
                "summary": "All target and retained tests passed.",
                "metadata": metadata,
            },
            role="evaluator",
        )

        return EvaluationResult(
            status="PASS",
            summary="All target and retained tests passed.",
            pass_to_fail_tests=len(task.target_tests),
            retained_tests=len(task.retained_tests),
            regression_detected=False,
            infrastructure_failure=False,
        )

    def _finalize_failure(
        self,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        run_id: str,
        task: EvaluationTask,
        status: str,
        summary: str,
        *,
        target_tests: list[str],
        retained_tests: list[str],
        target_run: dict[str, Any] | None = None,
        retained_run: dict[str, Any] | None = None,
        regression_detected: bool = False,
    ) -> EvaluationResult:
        metadata = {
            "task_id": task.task_id,
            "commit_sha": task.commit_sha,
            "status": status,
            "summary": summary,
            "target_tests": target_tests,
            "retained_tests": retained_tests,
            "target_run": target_run,
            "retained_run": retained_run,
            "artifacts": artifact_store.list(),
        }

        self._store_event(
            event_store,
            run_id,
            "EvaluationCompleted",
            {
                "status": status,
                "summary": summary,
                "metadata": metadata,
            },
            role="evaluator",
        )

        return EvaluationResult(
            status=status,
            summary=summary,
            pass_to_fail_tests=len(target_tests),
            retained_tests=len(retained_tests),
            regression_detected=regression_detected,
            infrastructure_failure=status == "INFRASTRUCTURE_FAILURE",
        )
