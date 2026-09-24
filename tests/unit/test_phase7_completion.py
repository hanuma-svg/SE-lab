from __future__ import annotations

import json
from pathlib import Path

from se_lab.cli import main as cli_main
from se_lab.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    ExperimentStore,
    wilson_interval,
)
from se_lab.reporting.dashboard import write_experiment_dashboard
from tests.unit.test_phase7_experiments import _fixture_config


def test_wilson_interval_is_bounded_and_deterministic():
    first = wilson_interval(2, 3)
    second = wilson_interval(2, 3)
    assert first == second
    assert 0 <= first[0] <= first[1] <= 1


def test_experiment_store_round_trip(tmp_path: Path):
    config_path = _fixture_config(tmp_path)
    config = ExperimentConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    result = ExperimentRunner().run(config)
    store = ExperimentStore(tmp_path / "registry.sqlite3")
    store.save(result)
    loaded = store.get(result.experiment_id)
    assert loaded.model_dump(mode="json") == result.model_dump(mode="json")
    assert store.list()[0]["experiment_id"] == result.experiment_id


def test_experiment_store_rejects_mutation_of_existing_id(tmp_path: Path):
    config_path = _fixture_config(tmp_path)
    config = ExperimentConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    result = ExperimentRunner().run(config)
    store = ExperimentStore(tmp_path / "registry.sqlite3")
    store.save(result)
    mutated = result.model_copy(update={"config_hash": "tampered"})
    try:
        store.save(mutated)
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("registry must reject mutation of an existing experiment ID")


def test_dashboard_contains_research_sections(tmp_path: Path):
    config_path = _fixture_config(tmp_path)
    config = ExperimentConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    result = ExperimentRunner().run(config)
    output = write_experiment_dashboard(result, tmp_path / "dashboard" / "index.html")
    content = output.read_text(encoding="utf-8")
    assert output.exists()
    assert result.experiment_id in content
    assert "Pass@1" in content
    assert "95% Wilson CI" in content
    assert "descriptive" in content.lower()
    assert "Machine-readable report" in content


def test_dashboard_and_registry_cli(tmp_path: Path, capsys):
    config_path = _fixture_config(tmp_path)
    result_path = tmp_path / "result.json"
    store_path = tmp_path / "registry.sqlite3"
    assert cli_main(["experiment", "--config", str(config_path), "--output", str(result_path), "--store", str(store_path)]) == 0
    capsys.readouterr()
    dashboard_path = tmp_path / "index.html"
    assert cli_main(["dashboard", "--experiment-result", str(result_path), "--output", str(dashboard_path)]) == 0
    dashboard_output = json.loads(capsys.readouterr().out)
    assert dashboard_output["status"] == "PASS"
    assert dashboard_path.exists()
    assert cli_main(["registry", "--store", str(store_path)]) == 0
    registry = json.loads(capsys.readouterr().out)
    assert registry[0]["experiment_id"] == "exp-smoke"
