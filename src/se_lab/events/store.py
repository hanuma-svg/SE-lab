from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from se_lab.contracts import EventEnvelope


class EventStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, event: EventEnvelope) -> EventEnvelope:
        sequence = self._next_sequence(event.run_id)
        timestamp = datetime.now(UTC).isoformat()
        event_id = event.event_id or str(uuid4())
        enriched = event.with_metadata(sequence=sequence, timestamp=timestamp, event_id=event_id)

        event_path = self.root / f"{event.run_id}.jsonl"
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched.model_dump(mode="json"), sort_keys=True) + "\n")

        return enriched

    def _next_sequence(self, run_id: str) -> int:
        event_path = self.root / f"{run_id}.jsonl"
        if not event_path.exists():
            return 1

        with event_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        if not lines:
            return 1

        last = json.loads(lines[-1])
        return int(last.get("sequence", 0)) + 1

    def read(self, run_id: str) -> list[EventEnvelope]:
        event_path = self.root / f"{run_id}.jsonl"
        if not event_path.exists():
            return []

        events: list[EventEnvelope] = []
        with event_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                events.append(EventEnvelope(**payload))

        return events
