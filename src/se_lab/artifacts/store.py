from __future__ import annotations

from hashlib import sha256
from pathlib import Path


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, object]] = {}

    def store_bytes(self, content: bytes, object_type: str) -> dict[str, object]:
        digest = sha256(content).hexdigest()
        if digest in self._records:
            return dict(self._records[digest])

        shard_dir = self.root / "sha256" / digest[:2]
        shard_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = shard_dir / digest

        if not artifact_path.exists():
            artifact_path.write_bytes(content)

        record = {
            "sha256": digest,
            "bytes": len(content),
            "object_type": object_type,
            "path": str(artifact_path),
        }
        self._records[digest] = record
        return dict(record)

    def list(self) -> list[dict[str, object]]:
        if self._records:
            return [dict(record) for record in self._records.values()]

        records: list[dict[str, object]] = []
        sha_root = self.root / "sha256"
        if not sha_root.exists():
            return records

        for digest_dir in sorted(sha_root.glob("*")):
            if not digest_dir.is_dir():
                continue
            for digest_file in sorted(digest_dir.glob("*")):
                if not digest_file.is_file():
                    continue
                records.append(
                    {
                        "sha256": digest_file.name,
                        "bytes": digest_file.stat().st_size,
                        "object_type": "artifact",
                        "path": str(digest_file),
                    }
                )

        return records

    def load_bytes(self, sha256_hex: str) -> bytes:
        artifact_path = self.root / "sha256" / sha256_hex[:2] / sha256_hex
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact {sha256_hex} not found")

        content = artifact_path.read_bytes()
        actual_digest = sha256(content).hexdigest()
        if actual_digest != sha256_hex:
            raise ValueError(
                f"Artifact integrity check failed for {sha256_hex}: stored content digest does not match requested digest."
            )
        return content
