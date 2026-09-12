from __future__ import annotations

from hashlib import sha256
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ModelRequest(BaseModel):
    task_id: str
    repository_path: str
    commit_sha: str
    prompt: str
    seed: int = 0
    provider_name: str = "mock"
    provider_version: str = "mock-v1"
    model_name: str = "mock-baseline"
    max_tokens: int | None = None
    max_model_calls: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    content: str
    provider_name: str
    provider_version: str
    model_name: str
    prompt_hash: str
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str = "stop"


@runtime_checkable
class ModelProvider(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse:
        ...


class ModelProviderError(RuntimeError):
    pass


class MockModelProvider:
    def __init__(self, *, seed: int = 0, response_text: str | None = None):
        self.seed = seed
        self.response_text = response_text

    def complete(self, request: ModelRequest) -> ModelResponse:
        if self.response_text is not None:
            content = self.response_text
        else:
            content = request.metadata.get("mock_patch") or ""

        if content == "":
            raise ModelProviderError("Mock model provider did not receive a valid offline patch response.")

        prompt_hash = sha256(request.prompt.encode("utf-8")).hexdigest()
        usage = {"prompt_tokens": len(request.prompt.split()), "completion_tokens": len(content.split())}

        return ModelResponse(
            content=content,
            provider_name=request.provider_name,
            provider_version=request.provider_version,
            model_name=request.model_name,
            prompt_hash=prompt_hash,
            usage=usage,
            finish_reason="stop",
        )
