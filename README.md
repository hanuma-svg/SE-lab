# SE-Lab

**Multi-Agent Software Engineering Evaluation Lab**

SE-Lab is an evaluation framework for comparing software-engineering agents under controlled, auditable conditions.

> **Measure the agent, not just the final patch.**

The system records agent actions, tool decisions, handoffs, artifacts, tests, policy decisions, failures, and replay evidence so experiments can be inspected after execution.

## What is implemented

- **Measurement spine** — typed run/event contracts, append-only JSONL evidence, and content-addressed artifacts
- **Policy gateway** — explicit allow/deny decisions for repository operations and sensitive actions
- **Independent evaluator** — evaluates candidate patches externally from agent internals
- **Single-agent baseline** — deterministic offline baseline for comparison
- **Bounded multi-agent workflow** — Planner → Implementer → Tester → Reviewer
- **Record/replay** — normalized request identity, evidence validation, mismatch detection, and replay without live execution
- **Evaluation lab** — quality, safety, coordination, evidence, efficiency, and failure classification
- **Experiment workflow** — matched runs, repetitions, budgets, benchmark catalogs, aggregation, and comparison
- **Research statistics** — Wilson 95% intervals, paired pass differences, failure distributions, and descriptive latency summaries
- **Durable registry** — SQLite experiment metadata and run index while evidence remains content-addressed on disk
- **Static dashboard** — self-contained HTML report for recruiter/demo review without a backend service
- **Provider abstraction** — deterministic mock provider plus an OpenAI-compatible provider with bounded calls and token accounting
- **Security verification** — evaluator/test-oracle integrity, path containment, policy-denial, and Docker worker tests

## Phase 4 multi-agent workflow

The bounded workflow is intentionally small:

```text
Planner → Implementer → Tester → Reviewer
```

The workflow:

- uses the `ModelProvider` abstraction for model access
- supports the deterministic `MockModelProvider` for offline experiments
- supports bounded OpenAI-compatible provider calls
- passes structured planner context into implementation
- routes tool actions through the policy gateway
- records role transitions, model interactions, tool actions, handoffs, and artifacts
- invokes the independent evaluator rather than allowing the agent to grade itself
- enforces bounded model-call, tool-call, retry, token, and wall-clock budgets where configured

### CLI usage

Run the multi-agent workflow against a task definition:

```bash
se-lab multi-agent --task path/to/task.json
```

Optional controls include:

- `--seed`
- `--max-model-calls`
- `--max-tool-calls`
- `--max-wall-clock`
- `--max-retries`
- `--provider-name`
- `--provider-version`
- `--model-name`
- custom event and artifact directories

The offline mock provider remains the reproducible path used by the recorded benchmark campaign.

## Phase 6 independent evaluation

Phase 6 consumes only externally observable `EventEnvelope` records and artifact hashes. It does not import agent or model implementations. The audit reports separate quality, safety, coordination, evidence, and efficiency metrics and never combines them into an arbitrary weighted score.

Run an audit for a recorded run:

```bash
se-lab phase6 --run-id RUN_ID --events-dir .se-lab/events --artifacts-dir .se-lab/artifacts
```

Evidence validation is fail-closed: malformed events, missing causal parents, non-contiguous sequences, missing terminal evidence, and corrupted referenced artifacts are classified as `INSUFFICIENT_EVIDENCE` and cannot produce an unqualified `PASS`.

## Phase 7 researcher workflow

The researcher workflow supports matched baseline/treatment experiments with repetitions, seeds, budgets, benchmark catalogs, and explicit ablations. The existing offline event and artifact stores remain the default. A configuration selects a task definition, baseline and/or multi-agent variants, repetitions, seed, matched budgets, and an explicit ablation. The supported meaningful ablation is `no_retries`; unsupported graph changes are reported as mismatches rather than silently simulated.

Run an experiment, inspect a task catalog, or compare two recorded experiment results:

```bash
se-lab experiment --config experiment.json --output result.json
se-lab benchmark validate --catalog benchmarks/frozen_smoke_suite.json
se-lab compare --left baseline-result.json --right treatment-result.json
se-lab dashboard --experiment-result result.json --output dashboard/index.html
se-lab registry --store .se-lab/experiments.sqlite3
```

Experiment output reports sample counts, pass rates, failure classes, efficiency metrics, budget validity, and mismatches. It does not produce an arbitrary aggregate score or claim that one workflow is superior. The recorded six-run campaign uses the deterministic mock provider and validates the experiment/evidence pipeline rather than real-world LLM performance.

The experiment command now persists a queryable registry entry by default:

```bash
se-lab experiment --config experiment.json --output result.json --store .se-lab/experiments.sqlite3
```

The generated dashboard includes variant pass rates, Wilson confidence intervals, model/tool usage, wall-clock summaries, budget validity, and the scientific caveat that small descriptive samples do not establish superiority. It is intentionally a static artifact so the result can be published with the experiment bundle.

### Scope boundaries

This repository intentionally does not include:

- LangGraph
- PostgreSQL or MinIO
- web dashboard features
- **production-grade external provider qualification**
- PostgreSQL/MinIO deployment adapters; SQLite is the current local durable registry
- SWE-bench evaluation harnesses
- additional agent roles beyond the current bounded four-role workflow

The goal is to provide a credible, auditable evaluation foundation while keeping the current workflow deliberately minimal.

## Continuous integration

GitHub Actions runs Ruff, the full Python test suite, package build checks, and the Docker security tests on Ubuntu. Docker verification is kept as a separate job so a missing local Docker daemon does not hide security regressions in CI.

## Web console

The repository also includes a thin public-demo surface: a React/TypeScript Vite console calls a small FastAPI REST API, which reuses the existing benchmark catalog, bounded workflows, policy gateway, evaluator, experiment runner, statistics, and SQLite registry. The API exposes only the three predefined frozen smoke tasks and the deterministic mock provider; it does not accept arbitrary commands, repositories, paths, uploads, or provider credentials.

### Live demo

The current sandbox deployment is available at [SE-Lab Experiment Console](https://8000-ixc5jcy5cxy08i5qhknvw-8f7d8073.sg2.manus.computer/). It is a temporary public service URL for demonstration and verification, not a production deployment. The public process uses local SQLite persistence; history should therefore be treated as deployment-local and non-durable if the sandbox is recycled.

### Live demo

The verified sandbox deployment is available at [SE-Lab Experiment Console](https://8000-ixc5jcy5cxy08i5qhknvw-8f7d8073.sg2.manus.computer). This URL is a temporary public sandbox service rather than a durable production deployment; its SQLite history is tied to the active service filesystem.

### Local setup

```bash
python3 -m pip install -e '.[web]'
cd web
npm install
npm run build
cd ..
uvicorn se_lab.web_api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. During frontend development, run `npm run dev` from `web/`; Vite proxies `/api` requests to the FastAPI server on port 8000.

### Web architecture

```text
React / TypeScript
        ↓ REST
FastAPI
        ↓
SE-Lab engine
        ↓
Policy + execution → evidence → independent evaluation
        ↓
Statistics + SQLite registry
```

The public demo uses a deterministic/mock provider, a curated smoke benchmark, bounded execution, and descriptive statistics. SQLite persistence is local filesystem persistence; deployment environments with ephemeral filesystems must not be described as durable history.
