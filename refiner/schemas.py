"""Typed contracts for the standalone FaithfulMed Refiner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VerifierReport(BaseModel):
    """Verifier contract used by the existing FaithfulMed scaffold."""

    model_config = ConfigDict(extra="forbid")

    faithful: bool = True
    unsupported_claims: list[str] = Field(default_factory=list)
    omissions: list[str] = Field(default_factory=list)
    reading_level_ok: bool = True

    @field_validator("unsupported_claims", "omissions")
    @classmethod
    def validate_issues(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            value = value.strip()
            if not value:
                raise ValueError("flagged issues must not be blank")
            if value not in cleaned:
                cleaned.append(value)
        return cleaned

    @property
    def flagged_issues(self) -> list[str]:
        """Return only explicit content flags; booleans do not authorize edits."""

        return [*self.unsupported_claims, *self.omissions]


class RefinerInput(BaseModel):
    """One independent, verifier-directed refinement request."""

    model_config = ConfigDict(extra="forbid")

    example_id: str = "standalone"
    source_text: str
    draft: str
    verifier_report: VerifierReport = Field(default_factory=VerifierReport)

    @field_validator("example_id", "source_text", "draft")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value


class RefinerOutput(BaseModel):
    """Minimal user-facing output contract."""

    model_config = ConfigDict(extra="forbid")

    needs_refinement: bool
    refined_text: str
    unresolved_issues: list[str] = Field(default_factory=list)

    @field_validator("refined_text")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("refined_text must not be empty")
        return value

    @field_validator("unresolved_issues")
    @classmethod
    def normalize_unresolved(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            value = value.strip()
            if not value:
                raise ValueError("unresolved issues must not be blank")
            if value not in cleaned:
                cleaned.append(value)
        return cleaned

    @model_validator(mode="after")
    def unresolved_requires_review(self) -> "RefinerOutput":
        if self.unresolved_issues and not self.needs_refinement:
            raise ValueError("unresolved issues require needs_refinement=true")
        return self


class EvaluationFixture(RefinerInput):
    """Gold annotation fields. These are never sent to the model."""

    gold_needs_refinement: bool
    category: str
    gold_flagged_issues: list[str] = Field(default_factory=list)
    expected_corrections: list[str] = Field(default_factory=list)
    gold_refined_text: str
    preserve_facts: list[str] = Field(default_factory=list)
    expected_unresolved_issues: list[str] = Field(default_factory=list)
    annotation_provenance: str


class PredictionRecord(BaseModel):
    """Per-example trace written by the evaluation command."""

    example_id: str
    category: str
    source_text: str
    draft: str
    verifier_report: VerifierReport
    gold_needs_refinement: bool
    gold_refined_text: str = Field(exclude=True)
    prediction: RefinerOutput | None = None
    error: str | None = None
    latency_seconds: float
    token_usage: dict[str, Any] | None = None
    correction_success: bool | None = None
    false_flags_rejected: int = 0
    false_flags_total: int = 0
    required_corrections_met: int = 0
    required_corrections_total: int = 0
    preserve_facts_retained: int = 0
    preserve_facts_total: int = 0
    readability_before: dict[str, float | int] = Field(default_factory=dict)
    readability_after: dict[str, float | int] = Field(default_factory=dict)
