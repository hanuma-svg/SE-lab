# Phase 7 Completion Audit

## Verdict

**PASS WITH LIMITATIONS**

The remaining high-impact local implementation slice is complete. The repository now includes durable experiment metadata, reproducible descriptive statistics, a self-contained dashboard artifact, package-build verification, and GitHub Actions CI. No commit or push was performed.

## Implemented

- Added deterministic Wilson 95% confidence intervals.
- Added paired baseline/treatment pass-at-1 differences keyed by repetition and ablation.
- Added mean and median wall-clock summaries plus failure-class distributions.
- Replaced the experiment runner's ad hoc aggregation with the statistics module.
- Added `ExperimentStore`, a SQLite-backed experiment registry with experiment and run tables plus indexes.
- Added `--store` persistence to the `experiment` CLI command.
- Added `registry --store ...` to list durable experiment metadata.
- Added `dashboard --experiment-result ... --output ...` to generate a self-contained HTML dashboard.
- Added dashboard sections for pass rate, Wilson intervals, model/tool usage, latency, budget validity, and scientific caveats.
- Added GitHub Actions quality and Docker-security jobs.
- Added package build verification through the `build` development dependency.
- Updated README documentation.
- Added four integration tests for statistics, SQLite round trips, dashboard generation, and CLI integration.

## New commands

```bash
se-lab experiment --config experiment.json --output result.json --store .se-lab/experiments.sqlite3
se-lab dashboard --experiment-result result.json --output dashboard/index.html
se-lab registry --store .se-lab/experiments.sqlite3
```

## Verification

```text
python3 -m pytest -q --ignore=tests/unit/test_phase2_docker.py
121 passed in 69.03s

ruff check .
All checks passed!

python3 -m build
Successfully built source distribution and wheel

git diff --check
Passed
```

## Full test limitation

The exact full command `python3 -m pytest -q` cannot pass in this sandbox because Docker is unavailable. Docker-specific tests fail while attempting to execute `docker --version` with:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'docker'
```

The CI workflow includes a dedicated Docker job that will run on a GitHub Actions Ubuntu runner with Docker available.

## Scope that remains intentionally incomplete

- PostgreSQL and MinIO production adapters are not included; SQLite is the local durable registry and existing content-addressed files remain the evidence store.
- The dashboard is a generated static HTML artifact, not a hosted live web application.
- The benchmark remains a curated smoke suite rather than SWE-bench scale.
- The campaign is descriptive; it does not claim multi-agent superiority.
- LangGraph, OpenTelemetry/Jaeger, and a production deployment are not introduced in this slice.

## Git status expectation

Modified and untracked files are intentionally left in the working tree for review. No commit and no push were performed.


# Hardening Audit Addendum — 2026-09-23

## Executive verdict

**PASS WITH LIMITATIONS.** The implementation audit found one genuine P1 issue in the new SQLite registry: saving a different result under an existing `experiment_id` previously overwrote historical metadata. That issue is fixed and covered by a regression test. No P0 issues remain.

## Implementation audit

| Component | Status | Evidence | Issue |
| --- | --- | --- | --- |
| Experiment runner | Implemented | `src/se_lab/experiments/runner.py`, Phase 7 tests | Single-task experiment scope remains a research limitation |
| SQLite registry | Implemented and hardened | `ExperimentStore`, round-trip and mutation-rejection tests | Local SQLite only; no PostgreSQL adapter |
| Statistics | Implemented | Wilson interval, paired difference, latency and failure summaries; tests pass | Descriptive intervals only; no causal significance claim |
| Dashboard | Implemented | Static HTML generation test and CLI smoke test pass | Generated artifact, not a hosted live application |
| Benchmark catalog | Implemented | Three-task catalog validation passes | Not SWE-bench scale |
| Security policy | Implemented | Existing policy/security tests pass; Docker tests require Docker | Container execution cannot be verified in this sandbox |
| CI | Implemented | `.github/workflows/ci.yml` includes Ruff, tests, build, and Docker job | GitHub-hosted execution still needs to run after review |
| Packaging | Implemented | `python3 -m build` succeeds | No P0/P1 packaging issue found |

## P0 issues

**None.**

## P1 issues fixed

1. **Mutable experiment registry records.** `ExperimentStore.save()` now rejects a different payload or configuration hash for an existing experiment ID and permits only an identical idempotent save.
2. **Default registry collisions during comparisons.** When `--store` is omitted and `--output` is supplied, the CLI derives an output-specific SQLite path, preventing two result files in one directory from overwriting one another's registry metadata.
3. **SQLite foreign-key enforcement.** Connections now enable `PRAGMA foreign_keys = ON`.

## P2 / future work

- PostgreSQL and MinIO adapters.
- Hosted API and live dashboard.
- LangGraph orchestration.
- OpenTelemetry and Jaeger.
- Multi-task experiment schema with explicit task IDs in paired statistical keys.
- Larger curated and held-out datasets.
- Three-seed real-provider campaign.
- Docker verification in GitHub Actions.
- Production deployment and access control.

## Verification

```text
python3 -m pytest -q tests/unit/test_phase7_completion.py
5 passed

python3 -m pytest -q tests/unit/test_phase7_experiments.py::test_experiment_and_compare_cli tests/unit/test_phase7_completion.py
6 passed

python3 -m pytest -q --ignore=tests/unit/test_phase2_docker.py
122 passed in 74.97s

ruff check .
All checks passed!

python3 -m build
Successfully built source distribution and wheel

git diff --check
Passed

python3 -m pytest -q tests/unit/test_phase2_docker.py
4 failed: Docker executable unavailable in the sandbox

se-lab benchmark validate --catalog benchmarks/frozen_smoke_suite.json
PASS; 3 tasks validated

se-lab demo
PASS; 6 recorded runs, 3/3 baseline and 3/3 multi-agent

se-lab dashboard --experiment-result research/initial_campaign/campaign-result.json --output /tmp/se-lab-dashboard.html
PASS; non-empty dashboard generated

se-lab registry --store .se-lab/experiments.sqlite3
PASS; registry read returned persisted experiments
```

## Git safety

No push, PR, reset, rewrite, or deletion was performed. The working tree contains only the intentional milestone changes and generated review artifacts. A local commit is recommended only after human review of the diff and after GitHub Actions confirms the Docker job.
