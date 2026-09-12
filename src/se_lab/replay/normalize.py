from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

VOLATILE_FIELD_NAMES = {
    "timestamp",
    "event_id",
    "run_id",
    "sequence",
    "record_id",
    "request_id",
    "artifact_id",
    "container_id",
    "process_id",
}

def normalize_request(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            if key in VOLATILE_FIELD_NAMES:
                continue
            normalized[key] = normalize_request(value[key])
        return normalized
    if isinstance(value, list):
        return [normalize_request(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_request(item) for item in value]
    if isinstance(value, str):
        return value
    return value


def compute_request_identity(
    operation_kind: str,
    request: dict[str, Any] | None,
    *,
    schema_version: str | None = None,
    workspace_snapshot_hash: str | None = None,
    policy_version: str | None = None,
    provider_context: dict[str, Any] | None = None,
    run_context: dict[str, Any] | None = None,
) -> str:
    normalized_request = normalize_request(request or {})

    if isinstance(normalized_request, dict):
        normalized_request.pop("schema_version", None)
        normalized_request.pop("workspace_snapshot_hash", None)
        normalized_request.pop("policy_version", None)

    request_schema_version = schema_version or (request or {}).get("schema_version") if isinstance(request, dict) else None
    request_workspace_snapshot_hash = workspace_snapshot_hash or (request or {}).get("workspace_snapshot_hash") if isinstance(request, dict) else None
    request_policy_version = policy_version or (request or {}).get("policy_version") if isinstance(request, dict) else None

    payload: dict[str, Any] = {
        "operation_kind": operation_kind,
        "request": normalized_request,
    }

    if request_schema_version is not None:
        payload["schema_version"] = request_schema_version
    if request_workspace_snapshot_hash is not None:
        payload["workspace_snapshot_hash"] = request_workspace_snapshot_hash
    if request_policy_version is not None:
        payload["policy_version"] = request_policy_version

    normalized_provider_context = normalize_request(provider_context or {})
    if normalized_provider_context:
        payload["provider_context"] = normalized_provider_context

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()
