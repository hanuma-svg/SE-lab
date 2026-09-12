from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    description: str = ""
    repository_path: str
    commit_sha: str
    target_tests: list[str] = Field(default_factory=list)
    retained_tests: list[str] = Field(default_factory=list)
    allowed_write_paths: list[str] = Field(default_factory=list)
    protected_paths: list[str] = Field(default_factory=list)
    difficulty: str = "smoke"
    category: str = "curated"
    expected_behavior: str = ""
    adversarial: bool = False
    mock_patch: str = ""
    provenance: str = Field(min_length=1)
    deterministic_setup: str = Field(min_length=1)


class BenchmarkCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "se-lab"
    version: str = "1"
    tasks: list[BenchmarkTask] = Field(default_factory=list)

    def task(self, task_id: str) -> BenchmarkTask:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"Unknown benchmark task: {task_id}")

    @classmethod
    def from_file(cls, path: str | Path) -> BenchmarkCatalog:
        source = Path(path)
        data: Any = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(data, list):
            data = {"tasks": data}
        return cls.model_validate(data)

    def write_to_file(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
