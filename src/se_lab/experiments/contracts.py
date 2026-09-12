from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MatchedBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_model_calls: int = Field(default=4, ge=1)
    max_tool_calls: int = Field(default=10, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)
    max_wall_clock: int = Field(default=300, ge=1)
    max_retries: int = Field(default=1, ge=0)


class AblationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["full", "no_reviewer", "no_retries", "no_handoffs"] = "full"
    enabled: bool = True


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(min_length=1)
    task_path: str = Field(min_length=1)
    variants: list[Literal["baseline", "multi_agent"]] = Field(default_factory=lambda: ["baseline", "multi_agent"])
    repetitions: int = Field(default=1, ge=1, le=100)
    seed: int = 0
    budget: MatchedBudget = Field(default_factory=MatchedBudget)
    ablations: list[AblationConfig] = Field(default_factory=lambda: [AblationConfig()])
    provider_name: str = "mock"
    provider_version: str = "mock-v1"
    model_name: str = "mock-baseline"
    evidence_dir: str = ".se-lab/experiments"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_variants(self) -> ExperimentConfig:
        if not self.variants:
            raise ValueError("At least one experiment variant is required")
        if len(set(self.variants)) != len(self.variants):
            raise ValueError("Experiment variants must be unique")
        if not self.ablations:
            raise ValueError("At least one ablation configuration is required")
        return self


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    run_id: str
    repetition: int = Field(ge=1)
    variant: str
    ablation: str
    seed: int
    status: str
    summary: str
    failure_class: str
    pass_at_1: bool
    model_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    retries: int = Field(ge=0)
    wall_clock_duration_seconds: float | None = Field(default=None, ge=0)
    budget_match: bool = True
    evidence_digest: str | None = None
    evidence_dir: str | None = None


class ExperimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    config_hash: str
    runs: list[RunRecord] = Field(default_factory=list)
    mismatches: list[str] = Field(default_factory=list)
    aggregation: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_matched_budgets(self) -> bool:
        return not self.mismatches and all(run.budget_match for run in self.runs)
