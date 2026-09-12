from __future__ import annotations

import json
from pathlib import Path

from se_lab.benchmarks import BenchmarkCatalog
from se_lab.cli import main as cli_main
from se_lab.evaluation import EvaluationTask, IndependentEvaluator

ROOT = Path(__file__).parents[2]


def _evaluate_smoke(name: str, tmp_path: Path):
    task_path = ROOT / "benchmarks" / "smoke_tasks" / f"{name}.json"
    task = EvaluationTask.from_file(task_path)
    patch_path = tmp_path / f"{name}.patch"
    patch_path.write_text(task.mock_patch, encoding="utf-8")
    return IndependentEvaluator().evaluate(task, patch_path)


def test_frozen_smoke_catalog_is_small_and_reviewable():
    catalog = BenchmarkCatalog.from_file(ROOT / "benchmarks" / "frozen_smoke_suite.json")
    assert len(catalog.tasks) == 3
    assert catalog.task("smoke-success-readme").adversarial is False
    assert catalog.task("smoke-protected-evaluator").adversarial is True
    assert catalog.task("smoke-path-traversal").category == "path-safety"


def test_frozen_smoke_tasks_exercise_success_and_security_boundaries(tmp_path):
    success = _evaluate_smoke("success_readme", tmp_path)
    protected = _evaluate_smoke("protected_evaluator", tmp_path)
    traversal = _evaluate_smoke("path_traversal", tmp_path)
    assert success.status == "PASS"
    assert protected.status == "FAIL"
    assert "protected evaluator asset" in protected.summary
    assert traversal.status == "INFRASTRUCTURE_FAILURE"
    assert "escape the workspace" in traversal.summary


def test_frozen_smoke_demo_runs_baseline_treatment_and_persists_evidence(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    output = tmp_path / "frozen-smoke-result.json"
    assert cli_main(["experiment", "--config", "configs/experiments/frozen_smoke_demo.json", "--output", str(output)]) == 0
    capsys.readouterr()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["aggregation"]["sample_count"] == 2
    assert {run["variant"] for run in result["runs"]} == {"baseline", "multi_agent"}
    assert all(run["evidence_dir"] for run in result["runs"])
    assert all(Path(run["evidence_dir"]).exists() for run in result["runs"])

    second_output = tmp_path / "frozen-smoke-result-2.json"
    assert cli_main(["experiment", "--config", "configs/experiments/frozen_smoke_demo.json", "--output", str(second_output)]) == 0
    capsys.readouterr()
    second = json.loads(second_output.read_text(encoding="utf-8"))
    assert result["config_hash"] == second["config_hash"]
    assert [run["run_id"] for run in result["runs"]] == [run["run_id"] for run in second["runs"]]
