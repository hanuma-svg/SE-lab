from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ReplayFidelity(str, Enum):
    EVENT_EVIDENCE = "event-evidence"
    DETERMINISTIC_OBSERVATION = "deterministic-observation"
    CONTEXT_VALIDATION = "context-validation"
    SIDE_EFFECT_FREE = "side-effect-free"
    RECONSTRUCTION = "reconstruction"


class ObservationRecord(BaseModel):
    record_id: str
    run_id: str
    operation_kind: str
    request_hash: str
    normalized_request: dict[str, Any] = Field(default_factory=dict)
    observation_payload: dict[str, Any] = Field(default_factory=dict)
    causal_parent: str | None = None
    sequence: int | None = None
    schema_version: str | None = None
    workspace_snapshot_hash: str | None = None
    policy_version: str | None = None
    provider_context: dict[str, Any] = Field(default_factory=dict)
    artifact_references: list[str] = Field(default_factory=list)
    integrity_hash: str | None = None

    def integrity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"integrity_hash"})
