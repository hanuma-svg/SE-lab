# SE-lab

SE-lab is an evaluation platform for deterministic software-engineering tasks. The current bounded scope includes:

- Phase 1: core contracts, artifact/event storage, replay, reporting, and CLI scaffolding
- Phase 2: secure workspace and policy gateway enforcement
- Phase 3: independent evaluator for deterministic, external validation
- Phase 4 (current): a minimal single-agent baseline that uses a provider abstraction and the policy gateway, but intentionally stays offline and non-agentic

## Single-agent baseline

The baseline is intentionally small and isolated:

- `SingleAgentBaseline` materializes a task repository into a controlled workspace
- `ModelProvider` provides an abstraction for model access
- `MockModelProvider` provides deterministic offline responses using `task.mock_patch`
- the baseline requests authorization via the policy gateway before emitting any write operations
- each run records events and artifacts, then invokes the independent evaluator for final scoring

### CLI usage

Run the baseline against a task definition:

```bash
se-lab baseline --task path/to/task.json
```

Optional flags:

- `--run-id`
- `--events-dir`
- `--artifacts-dir`
- `--seed`
- `--max-model-calls`
- `--max-wall-clock`
- `--max-retries`
- `--provider-name`
- `--provider-version`
- `--model-name`

### Offline test pattern

The baseline is designed for deterministic offline use. The built-in mock provider returns the task's `mock_patch`, so end-to-end tests can execute without external model APIs.

### Scope boundaries

This repository intentionally does not include:

- LangGraph or multi-agent orchestration
- planner/implementer/tester/reviewer agents
- PostgreSQL or MinIO
- web dashboard features
- external model API integrations
- SWE-bench evaluation harnesses

The goal is to provide a secure, credible foundation for future agentic work while keeping the current baseline deliberately minimal and auditable.
