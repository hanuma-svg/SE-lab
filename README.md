# SE-lab

SE-lab is an evaluation platform for deterministic software-engineering tasks. The current bounded scope includes:

- Phase 1: core contracts, artifact/event storage, replay, reporting, and CLI scaffolding
- Phase 2: secure workspace and policy gateway enforcement
- Phase 3: independent evaluator for deterministic, external validation
- Phase 4: a minimal bounded multi-agent workflow with Planner, Implementer, Tester, and Reviewer roles, using the provider abstraction and policy gateway while remaining offline and deterministic
- Phase 5: deterministic record/replay with normalized request identity and mismatch detection
- Phase 6: independent evaluation of quality, safety, coordination, evidence, efficiency, and failure classes
- Phase 7: local experiment configuration, benchmark catalogues, matched runs, one meaningful ablation, aggregation, and comparison

## Phase 4 multi-agent workflow

The current bounded workflow is intentionally small and isolated:

- `MultiAgentWorkflow` orchestrates a Planner → Implementer → Tester → Reviewer flow
- `ModelProvider` provides an abstraction for model access
- `MockModelProvider` provides deterministic offline responses using `task.mock_patch`
- each role is allowed only through the policy gateway and the shared workspace abstraction
- the workflow records events and artifacts for every role transition, then invokes the independent evaluator for final scoring

### CLI usage

Run the multi-agent workflow against a task definition:

```bash
se-lab multi-agent --task path/to/task.json
```

Optional flags:

- `--run-id`
- `--events-dir`
- `--artifacts-dir`
- `--seed`
- `--max-model-calls`
- `--max-tool-calls`
- `--max-wall-clock`
- `--max-retries`
- `--provider-name`
- `--provider-version`
- `--model-name`

### Offline test pattern

The workflow is designed for deterministic offline use. The built-in mock provider returns the task's `mock_patch`, so end-to-end tests can execute without external model APIs.

## Phase 6 evaluation audit

Phase 6 consumes only externally observable `EventEnvelope` records and artifact hashes. It does not import agent or model implementations. The audit reports separate quality, safety, coordination, evidence, and efficiency metrics and never combines them into an arbitrary weighted score.

Run an audit for a recorded run:

```bash
se-lab phase6 --run-id RUN_ID --events-dir .se-lab/events --artifacts-dir .se-lab/artifacts
```

Evidence validation is fail-closed: malformed events, missing causal parents, non-contiguous sequences, missing terminal evidence, and corrupted referenced artifacts are classified as `INSUFFICIENT_EVIDENCE` and cannot produce an unqualified `PASS`.

## Phase 7 researcher workflow

The local researcher workflow uses strict JSON contracts and keeps the existing offline event and artifact stores as the default. A configuration selects a task definition, baseline and/or multi-agent variants, repetitions, seed, matched budgets, and an explicit ablation. The supported meaningful ablation is `no_retries`; unsupported graph changes are reported as mismatches rather than silently simulated.

Run an experiment, inspect a task catalog, or compare two recorded experiment results:

```bash
se-lab experiment --config experiment.json --output result.json
se-lab benchmark --catalog benchmarks/smoke.json
se-lab compare --left baseline-result.json --right treatment-result.json
```

Experiment output remains descriptive. It reports sample counts, pass rates, failure classes, efficiency metrics, budget validity, and mismatches without claiming that one workflow is superior or producing an arbitrary aggregate score.

### Scope boundaries

This repository intentionally does not include:

- LangGraph
- PostgreSQL or MinIO
- web dashboard features
- external model API integrations
- SWE-bench evaluation harnesses
- additional agent roles beyond the current bounded four-role workflow

The goal is to provide a secure, credible foundation for future agentic work while keeping the current workflow deliberately minimal and auditable.
