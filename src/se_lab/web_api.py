from __future__ import annotations

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from se_lab.benchmarks import BenchmarkCatalog
from se_lab.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    ExperimentStore,
    MatchedBudget,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "benchmarks" / "frozen_smoke_suite.json"
REGISTRY_PATH = Path(os.getenv("SE_LAB_REGISTRY", str(ROOT / ".se-lab" / "experiments.sqlite3")))
EVIDENCE_ROOT = Path(os.getenv("SE_LAB_WEB_EVIDENCE", str(ROOT / ".se-lab" / "web-runs")))
TASK_FILES = {
    "smoke-success-readme": "success_readme.json",
    "smoke-protected-evaluator": "protected_evaluator.json",
    "smoke-path-traversal": "path_traversal.json",
}

app = FastAPI(title="SE-Lab API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("SE_LAB_CORS_ORIGINS", "*").split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="se-lab-run")
_lock = threading.Lock()
_jobs: dict[str, RunState] = {}


class RunRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=100)
    variant: Literal["baseline", "multi_agent", "both"] = "both"


class RunState(BaseModel):
    run_id: str
    experiment_id: str
    task_id: str
    variant: str
    status: Literal["QUEUED", "RUNNING", "EVALUATING", "COMPLETED", "FAILED"]
    result: dict[str, Any] | None = None
    error: str | None = None


def _catalog() -> BenchmarkCatalog:
    return BenchmarkCatalog.from_file(CATALOG_PATH)


def _task_payload(task: Any) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "description": task.description,
        "difficulty": task.difficulty,
        "category": task.category,
        "adversarial": task.adversarial,
        "target_tests": task.target_tests,
        "retained_tests": task.retained_tests,
        "allowed_write_paths": task.allowed_write_paths,
        "protected_paths": task.protected_paths,
    }


def _set_state(run_id: str, **changes: Any) -> None:
    with _lock:
        state = _jobs[run_id]
        _jobs[run_id] = state.model_copy(update=changes)


def _public_result(result: Any) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    payload.get("report", {}).pop("task_path", None)
    for run in payload.get("runs", []):
        run.pop("evidence_dir", None)
    for run in payload.get("report", {}).get("runs", []):
        run.pop("evidence_dir", None)
    return payload


def _execute(run_id: str, request: RunRequest) -> None:
    _set_state(run_id, status="RUNNING")
    task = _catalog().task(request.task_id)
    variants = ["baseline", "multi_agent"] if request.variant == "both" else [request.variant]
    task_path = ROOT / "benchmarks" / "smoke_tasks" / TASK_FILES[request.task_id]
    config = ExperimentConfig(
        experiment_id=run_id,
        task_path=str(task_path),
        variants=variants,
        repetitions=1,
        seed=0,
        budget=MatchedBudget(max_model_calls=4, max_tool_calls=30, max_wall_clock=30, max_retries=1),
        provider_name="mock",
        provider_version="mock-v1",
        model_name="mock-baseline",
        evidence_dir=str(EVIDENCE_ROOT),
        metadata={"surface": "public-demo", "task_id": task.task_id},
    )
    try:
        _set_state(run_id, status="EVALUATING")
        result = ExperimentRunner().run(config)
        ExperimentStore(REGISTRY_PATH).save(result)
        _set_state(run_id, status="COMPLETED", result=_public_result(result))
    except (KeyError, OSError, RuntimeError, ValueError):  # public API returns a safe failure without exposing paths or traces
        _set_state(run_id, status="FAILED", error="The bounded experiment failed before producing a result.")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "provider": "mock", "catalog": "frozen-smoke"}


@app.get("/api/tasks")
def tasks() -> dict[str, Any]:
    catalog = _catalog()
    return {"name": catalog.name, "version": catalog.version, "tasks": [_task_payload(task) for task in catalog.tasks]}


@app.get("/api/tasks/{task_id}")
def task(task_id: str) -> dict[str, Any]:
    try:
        return _task_payload(_catalog().task(task_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown benchmark task") from exc


@app.post("/api/runs", response_model=RunState, status_code=202)
def create_run(request: RunRequest) -> RunState:
    try:
        _catalog().task(request.task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown benchmark task") from exc
    run_id = f"web-{uuid.uuid4().hex[:12]}"
    state = RunState(run_id=run_id, experiment_id=run_id, task_id=request.task_id, variant=request.variant, status="QUEUED")
    with _lock:
        _jobs[run_id] = state
    _executor.submit(_execute, run_id, request)
    return state


@app.get("/api/runs/{run_id}", response_model=RunState)
def run_status(run_id: str) -> RunState:
    with _lock:
        state = _jobs.get(run_id)
    if state is None:
        try:
            result = ExperimentStore(REGISTRY_PATH).get(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        return RunState(run_id=run_id, experiment_id=run_id, task_id=str(result.report.get("task_id", "unknown")), variant=",".join(sorted({run.variant for run in result.runs})), status="COMPLETED", result=_public_result(result))
    return state


@app.get("/api/experiments")
def experiments() -> dict[str, Any]:
    entries = []
    for item in ExperimentStore(REGISTRY_PATH).list():
        try:
            result = ExperimentStore(REGISTRY_PATH).get(item["experiment_id"])
            item = {**item, "task_id": result.report.get("task_id"), "variants": sorted({run.variant for run in result.runs})}
        except KeyError:
            pass
        entries.append(item)
    return {"experiments": entries}


@app.get("/api/experiments/{experiment_id}")
def experiment(experiment_id: str) -> dict[str, Any]:
    try:
        return _public_result(ExperimentStore(REGISTRY_PATH).get(experiment_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc


@app.on_event("shutdown")
def shutdown() -> None:
    _executor.shutdown(wait=False, cancel_futures=True)


WEB_DIST = ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="frontend")
