from __future__ import annotations

from pydantic import BaseModel, Field


class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    workspace_snapshot_hash: str
    schema_version: str
