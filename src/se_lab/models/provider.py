from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
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


def request_identity(request: ModelRequest) -> str:
    canonical = json.dumps(
        {
            "task_id": request.task_id,
            "commit_sha": request.commit_sha,
            "prompt_hash": sha256(request.prompt.encode("utf-8")).hexdigest(),
            "seed": request.seed,
            "provider_name": request.provider_name,
            "provider_version": request.provider_version,
            "model_name": request.model_name,
            "max_tokens": request.max_tokens,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


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
        usage = {
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": len(content.split()),
            "total_tokens": len(request.prompt.split()) + len(content.split()),
        }

        return ModelResponse(
            content=content,
            provider_name=request.provider_name,
            provider_version=request.provider_version,
            model_name=request.model_name,
            prompt_hash=prompt_hash,
            usage=usage,
            finish_reason="stop",
        )


class OpenAICompatibleProvider:
    """Minimal OpenAI-compatible chat-completions adapter using the stdlib only."""

    def __init__(self, *, api_key: str, base_url: str, timeout_seconds: int = 60):
        if not api_key.strip():
            raise ModelProviderError("Real provider credentials are missing: set SE_LAB_API_KEY.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls) -> OpenAICompatibleProvider:
        return cls(
            api_key=os.environ.get("SE_LAB_API_KEY", ""),
            base_url=os.environ.get("SE_LAB_API_BASE", "https://api.openai.com/v1"),
            timeout_seconds=int(os.environ.get("SE_LAB_PROVIDER_TIMEOUT", "60")),
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        body = {
            "model": request.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "seed": request.seed,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        payload = json.dumps(body).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ModelProviderError(f"Real provider HTTP failure: {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise ModelProviderError(f"Real provider request failed: {type(exc).__name__}") from exc

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content:
                raise ValueError("empty model content")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelProviderError("Real provider returned a malformed response.") from exc
        usage = data.get("usage")
        if not isinstance(usage, dict):
            raise ModelProviderError("Real provider returned missing usage accounting.")
        try:
            normalized_usage = {key: int(usage[key]) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelProviderError("Real provider returned malformed usage accounting.") from exc
        if any(value < 0 for value in normalized_usage.values()) or normalized_usage["total_tokens"] < normalized_usage["prompt_tokens"] + normalized_usage["completion_tokens"]:
            raise ModelProviderError("Real provider returned impossible usage accounting.")
        return ModelResponse(
            content=content,
            provider_name=request.provider_name,
            provider_version=request.provider_version,
            model_name=request.model_name,
            prompt_hash=sha256(request.prompt.encode("utf-8")).hexdigest(),
            usage=normalized_usage,
            finish_reason=str(choice.get("finish_reason", "stop")),
        )


class BudgetedModelProvider:
    def __init__(
        self,
        provider: ModelProvider,
        *,
        max_calls: int,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
        max_total_tokens: int | None = None,
    ):
        self.provider = provider
        self.max_calls = max_calls
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        if self.calls >= self.max_calls:
            raise ModelProviderError("Model call budget exhausted before provider request.")
        estimated_input_tokens = len(request.prompt.split())
        if self.max_input_tokens is not None and estimated_input_tokens > self.max_input_tokens:
            raise ModelProviderError("Input-token budget exhausted before provider request.")
        remaining_output_tokens = None
        if self.max_total_tokens is not None:
            remaining_output_tokens = self.max_total_tokens - self.total_tokens - estimated_input_tokens
            if remaining_output_tokens <= 0:
                raise ModelProviderError("Total-token budget exhausted before provider request.")
        request_limit = request.max_tokens
        if self.max_output_tokens is not None:
            request_limit = min(request_limit, self.max_output_tokens) if request_limit is not None else self.max_output_tokens
        if remaining_output_tokens is not None:
            request_limit = min(request_limit, remaining_output_tokens) if request_limit is not None else remaining_output_tokens
        if request_limit != request.max_tokens:
            request = request.model_copy(update={"max_tokens": request_limit})
        response = self.provider.complete(request)
        usage = response.usage
        try:
            input_tokens = usage["prompt_tokens"]
            output_tokens = usage["completion_tokens"]
            total_tokens = usage["total_tokens"]
            if any(not isinstance(value, int) or isinstance(value, bool) for value in (input_tokens, output_tokens, total_tokens)):
                raise ValueError("non-integer usage")
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelProviderError("Provider returned missing or malformed usage accounting.") from exc
        if min(input_tokens, output_tokens, total_tokens) < 0 or total_tokens < input_tokens + output_tokens:
            raise ModelProviderError("Provider returned impossible usage accounting.")
        if self.max_input_tokens is not None and input_tokens > self.max_input_tokens:
            raise ModelProviderError("Input-token budget exhausted.")
        if self.max_output_tokens is not None and output_tokens > self.max_output_tokens:
            raise ModelProviderError("Output-token budget exhausted.")
        if self.max_total_tokens is not None and self.total_tokens + total_tokens > self.max_total_tokens:
            raise ModelProviderError("Total-token budget exhausted.")
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens
        return response


def create_provider(provider_name: str, *, task_patch: str | None = None) -> ModelProvider:
    if provider_name == "mock":
        return MockModelProvider(response_text=task_patch)
    if provider_name in {"openai", "openai-compatible"}:
        return OpenAICompatibleProvider.from_environment()
    raise ModelProviderError(f"Unsupported provider: {provider_name}")
