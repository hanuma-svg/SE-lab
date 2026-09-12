from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Workspace:
    run_id: str
    root: Path
    repository_path: Path
    allowed_write_paths: list[Path] = field(default_factory=list)
    resource_limits: dict[str, Any] = field(default_factory=dict)
    network_enabled: bool = False
    lifecycle_state: str = "created"

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.repository_path = Path(self.repository_path).resolve()
        self.allowed_write_paths = [Path(path).resolve() for path in self.allowed_write_paths]
