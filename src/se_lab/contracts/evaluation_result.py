from __future__ import annotations

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    status: str
    summary: str
    pass_to_fail_tests: int | None = None
    retained_tests: int | None = None
    regression_detected: bool | None = None
    infrastructure_failure: bool | None = None
