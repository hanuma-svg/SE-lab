from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from se_lab.experiments.contracts import ExperimentResult


class ExperimentStore:
    """Small durable registry; evidence payloads remain in content-addressed run directories."""

    def __init__(self, path: str | Path = ".se-lab/experiments.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    config_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                    variant TEXT NOT NULL,
                    repetition INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    failure_class TEXT NOT NULL,
                    pass_at_1 INTEGER NOT NULL,
                    evidence_dir TEXT,
                    UNIQUE(experiment_id, run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_runs_experiment_variant ON runs(experiment_id, variant);
                """
            )

    def save(self, result: ExperimentResult) -> None:
        payload = json.dumps(result.model_dump(mode="json"), sort_keys=True)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT config_hash, result_json FROM experiments WHERE experiment_id = ?",
                (result.experiment_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != result.config_hash or existing[1] != payload:
                    raise ValueError(
                        f"Experiment {result.experiment_id} already exists with different immutable evidence. "
                        "Choose a new experiment_id."
                    )
                return
            connection.execute(
                """
                INSERT INTO experiments(experiment_id, config_hash, status, sample_count, result_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.experiment_id,
                    result.config_hash,
                    "PASS" if not result.mismatches else "MISMATCH",
                    len(result.runs),
                    payload,
                ),
            )
            connection.execute("DELETE FROM runs WHERE experiment_id = ?", (result.experiment_id,))
            connection.executemany(
                """
                INSERT INTO runs(run_id, experiment_id, variant, repetition, seed, status, failure_class, pass_at_1, evidence_dir)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run.run_id,
                        result.experiment_id,
                        run.variant,
                        run.repetition,
                        run.seed,
                        run.status,
                        run.failure_class,
                        int(run.pass_at_1),
                        run.evidence_dir,
                    )
                    for run in result.runs
                ],
            )

    def get(self, experiment_id: str) -> ExperimentResult:
        with self._connect() as connection:
            row = connection.execute("SELECT result_json FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
        if row is None:
            raise KeyError(f"Experiment not found: {experiment_id}")
        return ExperimentResult.model_validate_json(row[0])

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT experiment_id, config_hash, status, sample_count, created_at FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "experiment_id": row[0],
                "config_hash": row[1],
                "status": row[2],
                "sample_count": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
