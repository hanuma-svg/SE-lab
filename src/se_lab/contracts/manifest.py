from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunManifest(BaseModel):
    experiment_id: str
    run_id: str
    task_id: str
    repository_url: str
    repository_commit_sha: str
    container_image_digest: str
    dependency_lock_hash: str
    model_provider: str
    model_version: str
    adapter_version: str
    system_prompt_hash: str
    role_prompt_hash: str
    tool_schema_version: str
    policy_version: str
    temperature: float
    sampling_parameters: dict[str, Any] = Field(default_factory=dict)
    seed: int
    maximum_model_calls: int
    maximum_tool_calls: int
    maximum_tokens: int
    maximum_wall_clock: int
    maximum_retries: int
    baseline_or_treatment: str
    artifact_retention_policy: str
    redaction_policy: str

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=True)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()

    def write_to_file(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json", exclude_none=True)
        payload["manifest_hash"] = self.content_hash()
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload["manifest_hash"]

    @classmethod
    def read_from_file(cls, path: str | Path) -> RunManifest:
        target = Path(path)
        data = json.loads(target.read_text(encoding="utf-8"))
        manifest_hash = data.pop("manifest_hash", None)
        expected_hash = sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if manifest_hash is None or manifest_hash != expected_hash:
            raise ValueError("Manifest integrity check failed")
        return cls(**data)
