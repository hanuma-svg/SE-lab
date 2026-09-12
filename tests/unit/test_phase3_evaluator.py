from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from se_lab.agents import MultiAgentWorkflow, SingleAgentBaseline
from se_lab.cli import main as cli_main
from se_lab.evaluation import EvaluationTask, IndependentEvaluator


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def _write_fixture_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "fixture_repo"
    repo.mkdir(parents=True, exist_ok=True)

    (repo / "src").mkdir(parents=True, exist_ok=True)
    (repo / "tests").mkdir(parents=True, exist_ok=True)

    (repo / "src" / "app.py").write_text(
        "def compute():\n    return 1\n",
        encoding="utf-8",
    )
    (repo / "src" / "other.py").write_text(
        "def retained_value():\n    return 'stable'\n",
        encoding="utf-8",
    )
    (repo / "src" / "se_lab").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "se_lab" / "evaluation").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "se_lab" / "evaluation" / "__init__.py").write_text(
        "# evaluator placeholder\n",
        encoding="utf-8",
    )

    (repo / "tests" / "test_target.py").write_text(
        "from src.app import compute\n\n\ndef test_target():\n    assert compute() == 2\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_retained.py").write_text(
        "from src.app import compute\nfrom src.other import retained_value\n\n\ndef test_retained():\n    assert compute() == 2\n    assert retained_value() == 'stable'\n",
        encoding="utf-8",
    )

    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.name", "Test User")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", "initial commit")

    commit_sha = _run(repo, "git", "rev-parse", "HEAD").stdout.strip()
    return repo, commit_sha


def _task_for(repo: Path, commit_sha: str, *, target_tests: list[str] | None = None, retained_tests: list[str] | None = None, max_wall_clock: int = 30) -> EvaluationTask:
    return EvaluationTask(
        task_id="fixture-task",
        repository_path=str(repo),
        commit_sha=commit_sha,
        target_tests=target_tests or ["tests/test_target.py"],
        retained_tests=retained_tests or ["tests/test_retained.py"],
        protected_paths=["src/se_lab/evaluation"],
        max_wall_clock=max_wall_clock,
        allowed_write_paths=[str(repo / "src")],
        description="deterministic evaluator fixture",
    )


def _write_patch(repo: Path, rel_path: str, new_content: str) -> Path:
    target = repo / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(new_content, encoding="utf-8")

    patch_path = repo.parent / f"{rel_path.replace('/', '_')}.patch"
    diff = _run(repo, "git", "diff", "--binary", "HEAD")
    patch_path.write_text(diff.stdout, encoding="utf-8")
    return patch_path


def _write_time_limit_fixture(repo: Path, commit_sha: str) -> tuple[EvaluationTask, Path]:
    timeout_test = repo / "tests" / "test_timeout.py"
    timeout_test.write_text(
        "import time\n\n\ndef test_timeout():\n    time.sleep(0.5)\n",
        encoding="utf-8",
    )

    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", "add timeout fixture")
    timeout_sha = _run(repo, "git", "rev-parse", "HEAD").stdout.strip()

    task = _task_for(repo, timeout_sha, target_tests=["tests/test_timeout.py"], retained_tests=[], max_wall_clock=0.1)

    patched_app = repo / "src" / "app.py"
    patched_app.write_text("def compute():\n    return 2\n", encoding="utf-8")
    patch_path = repo.parent / "timeout.patch"
    patch_path.write_text(_run(repo, "git", "diff", "--binary", "HEAD").stdout, encoding="utf-8")
    return task, patch_path


def test_evaluator_successful_patch_and_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)

    patch_path = _write_patch(repo, "src/app.py", "def compute():\n    return 2\n")

    eval_root = tmp_path / "eval-run-root"
    if eval_root.exists():
        shutil.rmtree(eval_root)
    monkeypatch.setattr(tempfile, "mkdtemp", lambda prefix="": str(eval_root))

    evaluator = IndependentEvaluator()
    result = evaluator.evaluate(task, patch_path)

    assert result.status == "PASS"
    assert result.regression_detected is False

    event_files = list((eval_root / "events").glob("*.jsonl"))
    assert event_files

    event_payloads = []
    for event_file in event_files:
        event_payloads.extend(json.loads(line) for line in event_file.read_text(encoding="utf-8").splitlines() if line.strip())

    assert {event["event_type"] for event in event_payloads} >= {"TestRunStarted", "TestRunCompleted", "ArtifactStored", "EvaluationCompleted"}
    assert (eval_root / "artifacts" / "sha256").exists()


def test_evaluator_rejects_invalid_patch(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)

    patch_path = tmp_path / "invalid.patch"
    patch_path.write_text("this is not a valid git patch\n", encoding="utf-8")

    result = IndependentEvaluator().evaluate(task, patch_path)

    assert result.status == "FAIL"
    assert "Malformed patch" in result.summary


def test_evaluator_rejects_protected_evaluator_asset(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)

    patch_path = _write_patch(
        repo,
        "src/se_lab/evaluation/__init__.py",
        "# evaluator placeholder\n# protected mutation\n",
    )

    result = IndependentEvaluator().evaluate(task, patch_path)

    assert result.status == "FAIL"
    assert "protected evaluator asset" in result.summary



def test_evaluator_detects_target_test_tampering(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)
    evaluator = IndependentEvaluator()

    oracle_snapshot = evaluator._snapshot_oracle(repo, task)

    target = repo / "tests" / "test_target.py"
    target.write_text("def test_target():\\n    assert True\\n", encoding="utf-8")

    changed = evaluator._verify_oracle_integrity(repo, task, oracle_snapshot)

    assert changed == ["tests/test_target.py"]


def test_evaluator_detects_retained_test_tampering(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)
    evaluator = IndependentEvaluator()

    oracle_snapshot = evaluator._snapshot_oracle(repo, task)

    retained = repo / "tests" / "test_retained.py"
    retained.write_text("def test_retained():\\n    assert True\\n", encoding="utf-8")

    changed = evaluator._verify_oracle_integrity(repo, task, oracle_snapshot)

    assert changed == ["tests/test_retained.py"]



def test_evaluator_detects_new_test_support_file(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)
    evaluator = IndependentEvaluator()

    oracle_snapshot = evaluator._snapshot_oracle(repo, task)

    (repo / "tests" / "conftest.py").write_text(
        "def pytest_configure(config):\n    pass\n",
        encoding="utf-8",
    )

    changed = evaluator._verify_oracle_integrity(repo, task, oracle_snapshot)

    assert changed == ["tests/conftest.py"]



def test_evaluator_does_not_run_tests_after_oracle_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)
    evaluator = IndependentEvaluator()

    patch_path = _write_patch(
        repo,
        "src/app.py",
        "def compute():\n    return 2\n",
    )

    original_verify = evaluator._verify_oracle_integrity

    def tamper_then_verify(repo_root, task_obj, before):
        target = repo_root / "tests" / "test_target.py"
        target.write_text(
            "def test_target():\n    assert True\n",
            encoding="utf-8",
        )
        return original_verify(repo_root, task_obj, before)

    monkeypatch.setattr(
        evaluator,
        "_verify_oracle_integrity",
        tamper_then_verify,
    )

    pytest_called = False

    def fail_if_tests_run(*args, **kwargs):
        nonlocal pytest_called
        pytest_called = True
        raise AssertionError("pytest must not run after oracle tampering")

    monkeypatch.setattr(evaluator, "_run_test_suite", fail_if_tests_run)

    result = evaluator.evaluate(task, patch_path)

    assert result.status == "INFRASTRUCTURE_FAILURE"
    assert "oracle integrity violation" in result.summary.lower()
    assert "tests/test_target.py" in result.summary
    assert pytest_called is False


def test_evaluator_detects_retained_regression(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)

    (repo / "src" / "app.py").write_text("def compute():\n    return 2\n", encoding="utf-8")
    (repo / "src" / "other.py").write_text("def retained_value():\n    return 'broken'\n", encoding="utf-8")

    patch_path = tmp_path / "retained_regression.patch"
    patch_path.write_text(_run(repo, "git", "diff", "--binary", "HEAD").stdout, encoding="utf-8")

    result = IndependentEvaluator().evaluate(task, patch_path)

    assert result.status == "FAIL"
    assert result.regression_detected is True
    assert "Retained tests regressed" in result.summary


def test_evaluator_reports_infrastructure_failure_for_missing_repository(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)
    task.repository_path = str(tmp_path / "missing_repo")

    patch_path = tmp_path / "missing_repo.patch"
    patch_path.write_text("diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@\n-return 1\n+return 2\n", encoding="utf-8")

    result = IndependentEvaluator().evaluate(task, patch_path)

    assert result.status == "INFRASTRUCTURE_FAILURE"


def test_evaluator_times_out_and_classifies_timeout(tmp_path: Path):
    repo, _ = _write_fixture_repo(tmp_path)
    task, patch_path = _write_time_limit_fixture(repo, _run(repo, "git", "rev-parse", "HEAD").stdout.strip())

    result = IndependentEvaluator().evaluate(task, patch_path)

    assert result.status == "TIMEOUT"
    assert "timed out" in result.summary.lower()


def test_cli_evaluate_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)

    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task.to_dict()), encoding="utf-8")

    patch_path = _write_patch(repo, "src/app.py", "def compute():\n    return 2\n")

    exit_code = cli_main(["evaluate", "--task", str(task_path), "--patch", str(patch_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "PASS"


def test_single_agent_baseline_runs_offline_and_records_artifacts(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)

    patch_path = _write_patch(repo, "src/app.py", "def compute():\n    return 2\n")
    task.mock_patch = patch_path.read_text(encoding="utf-8")

    result = SingleAgentBaseline().run(task)

    assert result.status == "PASS"
    assert result.evaluation is not None
    assert result.evaluation.status == "PASS"
    assert result.artifacts["candidate_patch"]
    assert result.artifacts["model_response"]
    assert any(event.event_type == "BaselineRunCompleted" for event in result.events)


def test_single_agent_baseline_rejects_policy_violations(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)

    patch_path = _write_patch(repo, "tests/test_target.py", "def test_target():\n    assert True\n")
    task.mock_patch = patch_path.read_text(encoding="utf-8")

    result = SingleAgentBaseline().run(task)

    assert result.status == "FAIL"
    assert "Patch rejected by policy" in result.summary
    assert any(event.event_type == "ToolAuthorizationDenied" for event in result.events)


def test_cli_baseline_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)

    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task.to_dict()), encoding="utf-8")

    patch_path = _write_patch(repo, "src/app.py", "def compute():\n    return 2\n")
    task.mock_patch = patch_path.read_text(encoding="utf-8")
    task_path.write_text(json.dumps(task.to_dict()), encoding="utf-8")

    exit_code = cli_main(["baseline", "--task", str(task_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "PASS"


def test_multi_agent_workflow_runs_offline_and_records_artifacts(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)

    patch_path = _write_patch(repo, "src/app.py", "def compute():\n    return 2\n")
    task.mock_patch = patch_path.read_text(encoding="utf-8")

    result = MultiAgentWorkflow().run(task)

    assert result.status == "PASS"
    assert result.evaluation is not None
    assert result.evaluation.status == "PASS"
    assert result.artifacts["candidate_patch"]
    assert any(event.event_type == "ReviewerDecisionRecorded" for event in result.events)
    assert result.budgets["model_calls_used"] == 2
    assert result.budgets["tool_calls_used"] >= 1


def test_multi_agent_workflow_handles_relative_allowed_write_paths(tmp_path: Path):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)
    task.allowed_write_paths = ["src"]

    patch_path = _write_patch(repo, "src/app.py", "def compute():\n    return 2\n")
    task.mock_patch = patch_path.read_text(encoding="utf-8")

    result = MultiAgentWorkflow().run(task)

    assert result.status == "PASS"
    assert result.evaluation is not None
    assert result.evaluation.status == "PASS"


def test_cli_multi_agent_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    repo, commit_sha = _write_fixture_repo(tmp_path)
    task = _task_for(repo, commit_sha)

    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task.to_dict()), encoding="utf-8")

    patch_path = _write_patch(repo, "src/app.py", "def compute():\n    return 2\n")
    task.mock_patch = patch_path.read_text(encoding="utf-8")
    task_path.write_text(json.dumps(task.to_dict()), encoding="utf-8")

    exit_code = cli_main(["multi-agent", "--task", str(task_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "PASS"


def test_evaluator_independence(tmp_path: Path):
    evaluator_source = (tmp_path / "evaluator.py")
    evaluator_source.write_text((Path("src/se_lab/evaluation/evaluator.py").read_text(encoding="utf-8")), encoding="utf-8")

    content = evaluator_source.read_text(encoding="utf-8")
    assert "se_lab.agents" not in content
    assert "LangGraph" not in content
    assert "model_provider" not in content
