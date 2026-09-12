from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from se_lab.artifacts.store import ArtifactStore
from se_lab.replay.contracts import ObservationRecord


class RecordStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def append_record(
        self,
        *,
        run_id: str,
        observation_payload: dict[str, object],
        operation_kind: str,
        normalized_request: dict[str, object],
        request_hash: str,
        causal_parent: str | None,
        sequence: int | None,
        schema_version: str,
        workspace_snapshot_hash: str | None,
        policy_version: str | None,
        provider_context: dict[str, object],
        artifact_references: list[str],
    ) -> ObservationRecord:
        record = ObservationRecord(
            record_id=str(uuid4()),
            run_id=run_id,
            operation_kind=operation_kind,
            request_hash=request_hash,
            normalized_request=normalized_request,
            observation_payload=observation_payload,
            causal_parent=causal_parent,
            sequence=sequence,
            schema_version=schema_version,
            workspace_snapshot_hash=workspace_snapshot_hash,
            policy_version=policy_version,
            provider_context=provider_context,
            artifact_references=artifact_references,
        )
        record.integrity_hash = self._compute_integrity_hash(record)
        self._write_record(record)
        return record

    def _write_record(self, record: ObservationRecord) -> None:
        record_path = self.root / f"{record.run_id}.jsonl"
        with record_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json", exclude_none=False), sort_keys=True) + "\n")

    def lookup_request(self, request_hash: str) -> ObservationRecord:
        for record in self.load_run("*"):
            if record.request_hash == request_hash:
                return record
        raise KeyError(f"No observation found for request_hash={request_hash}")

    def load_run(self, run_id: str) -> list[ObservationRecord]:
        if run_id == "*":
            records: list[ObservationRecord] = []
            for path in sorted(self.root.glob("*.jsonl")):
                records.extend(self._read_file(path))
            return self._deduplicate_records(records)

        path = self.root / f"{run_id}.jsonl"
        if not path.exists():
            return []
        return self._deduplicate_records(self._read_file(path))

    def _read_file(self, path: Path) -> list[ObservationRecord]:
        records: list[ObservationRecord] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Malformed record in {path} at line {line_number}: invalid JSON ({exc.msg})") from exc
                    try:
                        records.append(ObservationRecord(**payload))
                    except ValidationError as exc:
                        raise ValueError(f"Malformed record in {path} at line {line_number}: invalid ObservationRecord ({exc})") from exc
        except OSError as exc:
            raise ValueError(f"Malformed record in {path}: cannot read file ({exc})") from exc
        return records

    def _deduplicate_records(self, records: list[ObservationRecord]) -> list[ObservationRecord]:
        unique_records: list[ObservationRecord] = []
        group_by_request: dict[str, dict[str, ObservationRecord]] = {}

        for record in records:
            signature = self._record_signature(record)
            grouped = group_by_request.setdefault(record.request_hash, {})
            if signature in grouped:
                continue
            if grouped:
                raise ValueError(
                    f"Conflicting duplicate records detected for request_hash={record.request_hash}; multiple distinct observations were recorded for the same request."
                )
            grouped[signature] = record
            unique_records.append(record)

        return unique_records

    def _record_signature(self, record: ObservationRecord) -> str:
        payload = record.model_dump(
            mode="json",
            exclude={"record_id", "integrity_hash"},
        )
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def validate_record(self, record: ObservationRecord) -> bool:
        expected = self._compute_integrity_hash(record)
        return expected == record.integrity_hash

    def _compute_integrity_hash(self, record: ObservationRecord) -> str:
        payload = record.integrity_payload()
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        from hashlib import sha256

        return sha256(canonical.encode("utf-8")).hexdigest()

    def load_artifact_reference(self, artifact_store: ArtifactStore, reference: str) -> bytes:
        if reference.startswith("sha256:"):
            return artifact_store.load_bytes(reference.split(":", 1)[1])
        raise ValueError(f"Unsupported artifact reference: {reference}")
