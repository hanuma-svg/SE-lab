from __future__ import annotations

import json
from pathlib import Path

from se_lab.agents.baseline import BaselineConfig, SingleAgentBaseline
from se_lab.agents.multi_agent import MultiAgentConfig, MultiAgentWorkflow
from se_lab.evaluation import FailureClass, IndependentEvaluator, evaluate_events
from se_lab.models.provider import ModelProviderError
from tests.unit.test_phase3_evaluator import _task_for, _write_fixture_repo


class FailingProvider:
    def complete(self, request):
        raise ModelProviderError("Real provider HTTP failure: 429")


def test_truncated_candidate_remains_malformed(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)
    candidate = tmp_path / "truncated.patch"
    candidate.write_text(
        "diff --git a/README.md b/README.md\n--- a/README.md\n",
        encoding="utf-8",
    )

    result = IndependentEvaluator().evaluate(task, candidate)

    assert result.status == "FAIL"
    assert result.summary == "Malformed patch: no file paths found in patch."


def test_multi_agent_provider_failure_is_infrastructure_with_zero_calls(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    result = MultiAgentWorkflow().run(
        _task_for(repo, commit_sha),
        config=MultiAgentConfig(
            run_id="provider-429",
            provider=FailingProvider(),
            provider_name="openai-compatible",
            provider_version="openrouter",
            model_name="poolside/laguna-s-2.1:free",
            max_model_calls=4,
            max_retries=0,
        ),
    )

    audit = evaluate_events(result.run_id, result.events)
    assert result.status == "INFRASTRUCTURE_FAILURE"
    assert result.budgets["model_calls_used"] == 0
    assert "HTTP failure: 429" in result.summary
    assert audit.primary_failure == FailureClass.INFRASTRUCTURE_FAILURE


def test_baseline_provider_failure_is_infrastructure_with_zero_calls(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    result = SingleAgentBaseline().run(
        _task_for(repo, commit_sha),
        config=BaselineConfig(
            run_id="baseline-provider-429",
            provider=FailingProvider(),
            provider_name="openai-compatible",
            provider_version="openrouter",
            model_name="poolside/laguna-s-2.1:free",
            max_model_calls=4,
            max_retries=0,
        ),
    )

    audit = evaluate_events(result.run_id, result.events)
    assert result.status == "INFRASTRUCTURE_FAILURE"
    assert result.model_calls == 0
    assert "HTTP failure: 429" in result.summary
    assert FailureClass.INFRASTRUCTURE_FAILURE in audit.failure_classifications


def test_pilot_limits_match_approved_change():
    config = json.loads(
        Path("configs/experiments/real_provider_pilot.json").read_text(
            encoding="utf-8"
        )
    )
    budget = config["budget"]
    assert budget["max_input_tokens"] == 1600
    assert budget["max_output_tokens"] == 512
    assert budget["max_tokens"] == 512
    assert budget["max_total_tokens"] == 2112
    assert budget["max_retries"] == 0
    assert budget["max_wall_clock"] == 60
    assert config["model_name"] == "poolside/laguna-s-2.1:free"
    assert config["provider_name"] == "openai-compatible"
    assert config["variants"] == ["baseline", "multi_agent"]


def test_truncated_response_metadata_is_not_success():
    response = {
        "choices": [
            {
                "message": {"content": "diff --git a/README.md b/README.md"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 164, "completion_tokens": 256, "total_tokens": 420},
    }
    choice = response["choices"][0]
    assert choice["finish_reason"] == "length"
    assert response["usage"]["completion_tokens"] == 256
    assert "+++ " not in choice["message"]["content"]


def test_provider_error_does_not_expose_credentials():
    message = str(ModelProviderError("Real provider HTTP failure: 429"))
    assert "Authorization" not in message
    assert "Bearer" not in message
    assert "API_KEY" not in message
