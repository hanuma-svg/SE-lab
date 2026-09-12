from __future__ import annotations

import json
import urllib.error

import pytest

from se_lab.models.provider import (
    BudgetedModelProvider,
    MockModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    OpenAICompatibleProvider,
    create_provider,
    request_identity,
)


def _request() -> ModelRequest:
    return ModelRequest(
        task_id="task",
        repository_path=".",
        commit_sha="abc",
        prompt="return a patch",
        seed=3,
        provider_name="mock",
        provider_version="mock-v1",
        model_name="mock-model",
        max_tokens=4,
    )


def test_request_identity_is_deterministic_and_secret_free():
    first = request_identity(_request())
    second = request_identity(_request())
    assert first == second
    assert len(first) == 64
    assert "return a patch" not in first


def test_budgeted_provider_enforces_calls_and_total_tokens():
    provider = BudgetedModelProvider(
        MockModelProvider(response_text="patch"),
        max_calls=1,
        max_total_tokens=4,
    )
    response = provider.complete(_request())
    assert response.content == "patch"
    with pytest.raises(ModelProviderError, match="call budget"):
        provider.complete(_request())


def test_budgeted_provider_rejects_output_token_overage():
    class FakeProvider:
        def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                content="too much",
                provider_name="fake",
                provider_version="v1",
                model_name="fake",
                prompt_hash="x",
                usage={"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
            )

    with pytest.raises(ModelProviderError, match="Output-token budget"):
        BudgetedModelProvider(FakeProvider(), max_calls=1, max_output_tokens=2).complete(_request())


@pytest.mark.parametrize(
    "usage",
    [{"prompt_tokens": 1, "completion_tokens": 1}, {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0}, {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 3}],
)
def test_budgeted_provider_fails_closed_on_invalid_usage(usage):
    class FakeProvider:
        def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="patch", provider_name="fake", provider_version="v1", model_name="fake", prompt_hash="x", usage=usage)

    with pytest.raises(ModelProviderError, match="usage"):
        BudgetedModelProvider(FakeProvider(), max_calls=1).complete(_request())


def test_real_provider_missing_credentials_fails_before_request(monkeypatch):
    monkeypatch.delenv("SE_LAB_API_KEY", raising=False)
    with pytest.raises(ModelProviderError, match="SE_LAB_API_KEY"):
        create_provider("openai-compatible")


def test_real_provider_malformed_response_is_rejected(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": []}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    provider = OpenAICompatibleProvider(api_key="test-only", base_url="https://example.invalid")
    with pytest.raises(ModelProviderError, match="malformed"):
        provider.complete(_request())


def test_real_provider_success_normalizes_usage_without_logging_credentials(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "patch"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                }
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    response = OpenAICompatibleProvider(api_key="test-only", base_url="https://example.invalid").complete(_request())
    assert response.content == "patch"
    assert response.usage == {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}


def test_real_provider_sends_request_side_output_limit(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "patch"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}).encode()

    def capture(request, **kwargs):
        captured["body"] = json.loads(request.data)
        captured["authorization"] = request.headers["Authorization"]
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", capture)
    request = _request().model_copy(update={"max_tokens": 2})
    OpenAICompatibleProvider(api_key="test-only", base_url="https://example.invalid").complete(request)
    assert captured["body"]["max_tokens"] == 2
    assert captured["authorization"] == "Bearer test-only"


def test_real_provider_transport_failure_is_redacted(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.URLError("secret-looking transport detail")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    provider = OpenAICompatibleProvider(api_key="test-only", base_url="https://example.invalid")
    with pytest.raises(ModelProviderError, match="request failed") as error:
        provider.complete(_request())
    assert "secret-looking" not in str(error.value)


def test_real_provider_missing_usage_fails_closed(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "patch"}}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    provider = OpenAICompatibleProvider(api_key="test-only", base_url="https://example.invalid")
    with pytest.raises(ModelProviderError, match="usage"):
        provider.complete(_request())


def test_real_provider_factory_rejects_unknown_provider():
    with pytest.raises(ModelProviderError, match="Unsupported provider"):
        create_provider("unknown")
