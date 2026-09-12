from __future__ import annotations

from pydantic import BaseModel, Field


class PolicyDecision(BaseModel):
    decision: str
    capability: str
    reason: str
    request_id: str | None = None
    normalized_path: str | None = None
    policy_version: str | None = None
    limits: dict[str, object] = Field(default_factory=dict)
    decision_timestamp: str | None = None
    event_id: str | None = None
