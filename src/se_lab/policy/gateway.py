from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from se_lab.contracts import EventEnvelope, PolicyDecision, ToolRequest
from se_lab.events.store import EventStore
from se_lab.runtime.workspace import Workspace


class PolicyGateway:
    def __init__(self, event_store: EventStore, policy_version: str = "phase-2-policy-v1"):
        self.event_store = event_store
        self.policy_version = policy_version

    def authorize(self, request: ToolRequest, workspace: Workspace) -> PolicyDecision:
        if not request.tool_name or not isinstance(request.arguments, dict):
            return self._deny(
                request,
                "UNKNOWN",
                "Malformed tool request.",
                workspace=workspace,
            )

        normalized_path = self._normalize_path(request.arguments.get("path"), workspace)

        if request.tool_name == "read_file":
            capability = "READ_REPOSITORY"
            if normalized_path is None or not self._is_within_workspace(normalized_path, workspace.repository_path):
                return self._deny(
                    request,
                    capability,
                    "Repository read attempted outside declared workspace.",
                    workspace=workspace,
                    normalized_path=normalized_path,
                )
            return self._allow(
                request,
                capability,
                "Repository read allowed inside declared workspace.",
                workspace=workspace,
                normalized_path=normalized_path,
                limits={"workspace_root": str(workspace.root), "repository_path": str(workspace.repository_path)},
            )

        if request.tool_name == "write_file":
            capability = "WRITE_PATCH"
            if normalized_path is None or not self._is_allowed_write_path(normalized_path, workspace):
                return self._deny(
                    request,
                    capability,
                    "Patch write outside declared workspace or permitted path.",
                    workspace=workspace,
                    normalized_path=normalized_path,
                    limits={"allowed_write_paths": [str(path) for path in workspace.allowed_write_paths]},
                )
            return self._allow(
                request,
                capability,
                "Patch write allowed inside workspace.",
                workspace=workspace,
                normalized_path=normalized_path,
                limits={"allowed_write_paths": [str(path) for path in workspace.allowed_write_paths]},
            )

        if request.tool_name == "run_tests":
            return self._allow(
                request,
                "RUN_TESTS",
                "Test execution allowed with resource limits.",
                workspace=workspace,
                limits=workspace.resource_limits,
            )

        if request.tool_name == "run_shell":
            return self._deny(
                request,
                "RUN_SHELL",
                "Arbitrary shell execution is denied by default.",
                workspace=workspace,
                limits={"allowed_executables": ["pytest"]},
            )

        if request.tool_name == "network":
            return self._deny(
                request,
                "NETWORK",
                "Network access is denied by default.",
                workspace=workspace,
                limits={"network_enabled": False},
            )

        if request.tool_name == "read_env":
            return self._deny(
                request,
                "READ_ENV",
                "Environment variables are denied by default.",
                workspace=workspace,
                limits={"read_env": False},
            )

        if request.tool_name == "host_filesystem":
            return self._deny(
                request,
                "HOST_FILESYSTEM",
                "Host filesystem access is denied.",
                workspace=workspace,
                limits={"host_filesystem": False},
            )

        if request.tool_name == "docker_socket":
            return self._deny(
                request,
                "DOCKER_SOCKET",
                "Docker socket access is denied.",
                workspace=workspace,
                limits={"docker_socket": False},
            )

        if request.tool_name == "security_configuration":
            return self._deny(
                request,
                "SECURITY_CONFIGURATION",
                "Security configuration changes are denied.",
                workspace=workspace,
                limits={"security_configuration": False},
            )

        if request.tool_name == "delete_artifact":
            return self._deny(
                request,
                "DELETE_ARTIFACT",
                "Artifact deletion is denied by default.",
                workspace=workspace,
                limits={"artifact_deletion": False},
            )

        return self._deny(
            request,
            "UNKNOWN",
            "Unsupported or malformed tool request.",
            workspace=workspace,
            limits={
                "supported_tools": [
                    "read_file",
                    "write_file",
                    "run_tests",
                    "run_shell",
                    "network",
                    "read_env",
                    "host_filesystem",
                    "docker_socket",
                    "security_configuration",
                    "delete_artifact",
                ]
            },
        )

    def _allow(
        self,
        request: ToolRequest,
        capability: str,
        reason: str,
        *,
        workspace: Workspace,
        normalized_path: str | None = None,
        limits: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        decision = PolicyDecision(
            decision="allow",
            capability=capability,
            reason=reason,
            request_id=self._request_id(request, normalized_path),
            normalized_path=normalized_path,
            policy_version=self.policy_version,
            limits=limits or {},
            decision_timestamp=datetime.now(UTC).isoformat(),
        )
        self._record_authorization_event(workspace.run_id, request, decision)
        return decision

    def _deny(
        self,
        request: ToolRequest,
        capability: str,
        reason: str,
        *,
        workspace: Workspace,
        normalized_path: str | None = None,
        limits: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        decision = PolicyDecision(
            decision="deny",
            capability=capability,
            reason=reason,
            request_id=self._request_id(request, normalized_path),
            normalized_path=normalized_path,
            policy_version=self.policy_version,
            limits=limits or {},
            decision_timestamp=datetime.now(UTC).isoformat(),
        )
        self._record_authorization_event(workspace.run_id, request, decision)
        return decision

    def _record_authorization_event(self, run_id: str, request: ToolRequest, decision: PolicyDecision) -> None:
        self.event_store.append(
            EventEnvelope(
                run_id=run_id,
                event_type="ToolAuthorizationRequested",
                role="policy_gateway",
                payload={
                    "request": request.model_dump(mode="json"),
                    "decision": decision.model_dump(mode="json"),
                },
            )
        )

        if decision.decision == "deny":
            self.event_store.append(
                EventEnvelope(
                    run_id=run_id,
                    event_type="ToolAuthorizationDenied",
                    role="policy_gateway",
                    payload={
                        "request": request.model_dump(mode="json"),
                        "decision": decision.model_dump(mode="json"),
                    },
                )
            )
            self.event_store.append(
                EventEnvelope(
                    run_id=run_id,
                    event_type="PolicyViolationDetected",
                    role="policy_gateway",
                    payload={
                        "request": request.model_dump(mode="json"),
                        "decision": decision.model_dump(mode="json"),
                    },
                )
            )

    def _normalize_path(self, path: Any, workspace: Workspace) -> str | None:
        if not isinstance(path, str) or path is None or path == "" or "\x00" in path:
            return None

        candidate = Path(path)
        if candidate.is_absolute():
            return None
        if ".." in candidate.parts:
            return None

        candidate_path = (workspace.repository_path / candidate).resolve(strict=False)
        root_path = workspace.root.resolve(strict=False)
        if not str(candidate_path).startswith(str(root_path)):
            return None
        return str(candidate_path)

    def _is_within_workspace(self, normalized_path: str, repository_path: Path) -> bool:
        candidate = Path(normalized_path).resolve(strict=False)
        repository_root = repository_path.resolve(strict=False)
        return repository_root == candidate or repository_root in candidate.parents

    def _is_allowed_write_path(self, normalized_path: str, workspace: Workspace) -> bool:
        candidate = Path(normalized_path).resolve(strict=False)
        workspace_root = workspace.root.resolve(strict=False)
        if not str(candidate).startswith(str(workspace_root)):
            return False
        if not workspace.allowed_write_paths:
            return True
        for allowed in workspace.allowed_write_paths:
            allowed_path = Path(allowed).resolve(strict=False)
            if str(candidate).startswith(str(allowed_path)):
                return True
        return False

    def _request_id(self, request: ToolRequest, normalized_path: str | None) -> str:
        payload = {
            "tool_name": request.tool_name,
            "arguments": request.arguments,
            "normalized_path": normalized_path,
            "workspace_snapshot_hash": request.workspace_snapshot_hash,
            "schema_version": request.schema_version,
        }
        text = str(payload).encode("utf-8")
        return sha256(text).hexdigest()
