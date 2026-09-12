from __future__ import annotations

from pydantic import BaseModel


class PolicyDecision(BaseModel):
    decision: str
    capability: str
    reason: str
    event_id: str | None = None
