from __future__ import annotations

from pathlib import Path

from se_lab.artifacts.store import ArtifactStore
from se_lab.contracts import ToolRequest
from se_lab.events.store import EventStore
from se_lab.policy.command_policy import CommandPolicy
from se_lab.policy.gateway import PolicyGateway
from se_lab.runtime.workspace import Workspace


def _workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    repo = root / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    return Workspace(
        run_id="run-001",
        root=root,
        repository_path=repo,
        allowed_write_paths=[repo],
        resource_limits={"cpu": 1, "memory": "512m", "wall_clock": 60, "output_size": 1048576},
    )


def test_policy_allows_valid_repository_read(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="read_file",
        arguments={"path": "README.md"},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    decision = gateway.authorize(request, workspace)

    assert decision.decision == "allow"
    assert decision.capability == "READ_REPOSITORY"


def test_policy_allows_valid_patch_write_inside_workspace(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="write_file",
        arguments={"path": "repo/patch.txt"},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    decision = gateway.authorize(request, workspace)

    assert decision.decision == "allow"
    assert decision.capability == "WRITE_PATCH"


def test_policy_allows_test_execution(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="run_tests",
        arguments={"command": ["pytest", "-q"]},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    decision = gateway.authorize(request, workspace)

    assert decision.decision == "allow"
    assert decision.capability == "RUN_TESTS"


def test_policy_denies_parent_traversal(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="read_file",
        arguments={"path": "../../secret.txt"},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    decision = gateway.authorize(request, workspace)

    assert decision.decision == "deny"
    assert decision.capability == "READ_REPOSITORY"


def test_policy_denies_absolute_host_path(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="read_file",
        arguments={"path": "/etc/passwd"},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    decision = gateway.authorize(request, workspace)

    assert decision.decision == "deny"


def test_policy_denies_write_outside_workspace(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="write_file",
        arguments={"path": "/tmp/payload.txt"},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    decision = gateway.authorize(request, workspace)

    assert decision.decision == "deny"
    assert decision.capability == "WRITE_PATCH"


def test_policy_denies_network_capability(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="network",
        arguments={},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    decision = gateway.authorize(request, workspace)

    assert decision.decision == "deny"
    assert decision.capability == "NETWORK"


def test_policy_denies_environment_access(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="read_env",
        arguments={},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    decision = gateway.authorize(request, workspace)

    assert decision.decision == "deny"


def test_policy_denies_docker_socket_access(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="docker_socket",
        arguments={},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    decision = gateway.authorize(request, workspace)

    assert decision.decision == "deny"


def test_policy_denies_security_configuration_changes(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="security_configuration",
        arguments={},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    decision = gateway.authorize(request, workspace)

    assert decision.decision == "deny"


def test_policy_denies_artifact_deletion(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="delete_artifact",
        arguments={},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    decision = gateway.authorize(request, workspace)

    assert decision.decision == "deny"


def test_policy_denies_arbitrary_shell(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="run_shell",
        arguments={"command": "pytest && rm -rf /tmp/evil"},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    decision = gateway.authorize(request, workspace)

    assert decision.decision == "deny"


def test_malformed_tool_request_fails_closed(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="",
        arguments={},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    decision = gateway.authorize(request, workspace)

    assert decision.decision == "deny"


def test_policy_decision_records_authz_event(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="read_file",
        arguments={"path": "README.md"},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    gateway.authorize(request, workspace)
    events = event_store.read("run-001")

    assert any(event.event_type == "ToolAuthorizationRequested" for event in events)


def test_denied_request_creates_authorization_denied_event(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="run_shell",
        arguments={"command": "pytest && echo bad"},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    gateway.authorize(request, workspace)
    events = event_store.read("run-001")

    assert any(event.event_type == "ToolAuthorizationDenied" for event in events)
    assert any(event.event_type == "PolicyViolationDetected" for event in events)


def test_policy_decision_is_deterministic_for_identical_requests(tmp_path):
    workspace = _workspace(tmp_path)
    event_store = EventStore(tmp_path / "events")
    gateway = PolicyGateway(event_store)

    request = ToolRequest(
        tool_name="read_file",
        arguments={"path": "README.md"},
        workspace_snapshot_hash="abc",
        schema_version="v1",
    )

    first = gateway.authorize(request, workspace)
    second = gateway.authorize(request, workspace)

    assert first.decision == second.decision
    assert first.request_id == second.request_id


def test_command_policy_rejects_chaining_and_redirects():
    policy = CommandPolicy()

    assert policy.validate("pytest -q") is True
    assert policy.validate("pytest && echo bad") is False
    assert policy.validate("pytest > out.txt") is False
    assert policy.validate("bash -c 'echo hi'") is False


def test_artifact_store_persists_requested_artifacts(tmp_path):
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    artifact_store.store_bytes(b"payload", object_type="artifact")

    list_items = artifact_store.list()

    assert len(list_items) == 1
    assert list_items[0]["object_type"] == "artifact"
