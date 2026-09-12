from __future__ import annotations

from se_lab.artifacts.store import ArtifactStore
from se_lab.events.store import EventStore


def build_report(run_id: str, event_store: EventStore, artifact_store: ArtifactStore) -> dict[str, object]:
    events = event_store.read(run_id)
    artifacts = artifact_store.list()

    return {
        "run_id": run_id,
        "event_count": len(events),
        "events": [event.model_dump(mode="json") for event in events],
        "artifacts": artifacts,
    }
