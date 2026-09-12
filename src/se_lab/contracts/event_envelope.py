from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EventEnvelope(BaseModel):
    run_id: str
    event_type: str
    role: str
    payload: dict[str, Any] = Field(default_factory=dict)
    sequence: int | None = None
    event_id: str | None = None
    timestamp: str | None = None

    def with_metadata(self, sequence: int, timestamp: str, event_id: str) -> EventEnvelope:
        return EventEnvelope(
            run_id=self.run_id,
            event_type=self.event_type,
            role=self.role,
            payload=self.payload,
            sequence=sequence,
            event_id=event_id,
            timestamp=timestamp,
        )
