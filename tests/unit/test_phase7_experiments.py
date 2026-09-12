from __future__ import annotations

import json
from pathlib import Path

from se_lab.benchmarks import BenchmarkCatalog, BenchmarkTask
from se_lab.cli import main as cli_main
from se_lab.experiments import ExperimentConfig, ExperimentRunner
from tests.unit.test_phase3_evaluator import (
    _task_for,
    _write_fixture_repo,
    _write_patch,
)


def _fixture_config(tmp_path: Path) -> Path:
    repo, commit_sha = _write_fixture_repo(tmp_path)
    patch = _write_patch(repo, "src/app.py", "def compute():\n    return 2\n")
    task = _task_for(repo, commit_sha)
    task.mock_patch = patch.read_text(encoding="utf-8")
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task.to_dict()), encoding="utf-8")
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "exp-smoke",
                "task_path": str(task_path),
                "evidence_dir": str(tmp_path / "evidence"),
                "repetitions": 1,
                "seed": 11,
                "budget": {"max_model_calls": 4, "max_tool_calls": 30, "max_wall_clock": 30, "max_retries": 1},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_experiment_runner_executes_matched_baseline_and_treatment(tmp_path):
    config_path = _fixture_config(tmp_path)
    config = ExperimentConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    result = ExperimentRunner().run(config)

    assert result.mismatches == []
    assert result.has_matched_budgets is True
    assert {run.variant for run in result.runs} == {"baseline", "multi_agent"}
    assert all(run.pass_at_1 for run in result.runs)
    assert all(run.evidence_dir and (Path(run.evidence_dir) / "events").exists() for run in result.runs)
    assert all(run.evidence_dir and (Path(run.evidence_dir) / "artifacts").exists() for run in result.runs)
    assert result.aggregation["sample_count"] == 2
    assert result.report["matched_budget_valid"] is True


def test_multi_agent_implementer_receives_full_planner_context(tmp_path):
    config_path = _fixture_config(tmp_path)
    config = ExperimentConfig.model_validate_json(config_path.read_text(encoding="utf-8"))

    from se_lab.agents import MultiAgentWorkflow

    workflow = MultiAgentWorkflow()
    task = workflow._load_task(config.task_path)

    prompt = workflow._role_prompt(
        task,
        "implementer",
        {
            "planner_summary": "Implement the requested README change.",
            "plan_steps": ["inspect", "patch", "test", "review"],
            "affected_paths": ["README.md"],
            "expected_tests": ["tests/test_target.py"],
            "retained_tests": ["tests/test_retained.py"],
            "artifact_references": ["sha256:planner-artifact"],
            "revision_instructions": ["Preserve evaluator boundaries."],
        },
    )

    assert "Implement the requested README change." in prompt
    assert "inspect" in prompt
    assert "README.md" in prompt
    assert "tests/test_target.py" in prompt
    assert "tests/test_retained.py" in prompt
    assert "sha256:planner-artifact" in prompt
    assert "Preserve evaluator boundaries." in prompt
    assert "Return a valid unified git patch only." in prompt


def test_repeated_execution_fails_closed_without_mixing_evidence(tmp_path):
    config_path = _fixture_config(tmp_path)
    config = ExperimentConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    runner = ExperimentRunner()
    runner.run(config)

    try:
        runner.run(config)
    except FileExistsError as exc:
        assert "historical evidence was preserved" in str(exc)
    else:
        raise AssertionError("reusing an evidence directory must fail closed")


def test_no_retries_ablation_is_explicit_and_deterministic(tmp_path):
    config_path = _fixture_config(tmp_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["ablations"] = [{"name": "no_retries", "enabled": True}]
    config_path.write_text(json.dumps(data), encoding="utf-8")
    result = ExperimentRunner().run(ExperimentConfig.model_validate(data))
    assert result.mismatches == []
    assert all(run.ablation == "no_retries" for run in result.runs)
    assert all(run.retries == 0 for run in result.runs)


def test_unsupported_ablation_is_reported_not_silently_applied(tmp_path):
    config_path = _fixture_config(tmp_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["ablations"] = [{"name": "no_reviewer", "enabled": True}]
    result = ExperimentRunner().run(ExperimentConfig.model_validate(data))
    assert result.runs == []
    assert result.mismatches


def test_benchmark_catalog_and_cli(tmp_path, capsys):
    catalog = BenchmarkCatalog(
        tasks=[
            BenchmarkTask(
                task_id="task-1",
                repository_path="repo",
                commit_sha="sha",
                adversarial=True,
                provenance="unit fixture",
                deterministic_setup="fixed unit fixture",
            )
        ]
    )
    catalog_path = tmp_path / "catalog.json"
    catalog.write_to_file(catalog_path)
    assert cli_main(["benchmark", "--catalog", str(catalog_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["task_count"] == 1
    assert payload["adversarial_count"] == 1


def test_experiment_and_compare_cli(tmp_path, capsys):
    config_path = _fixture_config(tmp_path)
    second_config = json.loads(config_path.read_text(encoding="utf-8"))
    second_config["evidence_dir"] = str(tmp_path / "evidence-second")
    second_config_path = tmp_path / "experiment-second.json"
    second_config_path.write_text(json.dumps(second_config), encoding="utf-8")
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    assert cli_main(["experiment", "--config", str(config_path), "--output", str(left_path)]) == 0
    capsys.readouterr()
    assert cli_main(["experiment", "--config", str(second_config_path), "--output", str(right_path)]) == 0
    capsys.readouterr()
    assert cli_main(["compare", "--left", str(left_path), "--right", str(right_path)]) == 1
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["comparable"] is False
    assert "configuration hashes differ" in comparison["mismatches"][0].lower()
    assert cli_main(["report", "--experiment-result", str(left_path), "--kind", "failure"]) == 0
    failure_report = json.loads(capsys.readouterr().out)
    assert failure_report["report_type"] == "failure_analysis"


def test_experiment_config_rejects_empty_variants():
    try:
        ExperimentConfig(experiment_id="x", task_path="task", variants=[])
    except ValueError as exc:
        assert "variant" in str(exc).lower()
    else:
        raise AssertionError("empty variants must be rejected")
