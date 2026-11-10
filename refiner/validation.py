"""Deterministic parsing and invariant checks for Refiner responses."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .schemas import RefinerInput, RefinerOutput


class RefinerError(RuntimeError):
    """Base class for explicit Refiner operational failures."""


class RefinerConfigurationError(RefinerError):
    """The runtime is not configured for a requested live call."""


class RefinerResponseError(RefinerError):
    """The model returned malformed or contract-violating content."""


def unchanged_output(case: RefinerInput) -> RefinerOutput:
    return RefinerOutput(
        needs_refinement=False,
        refined_text=case.draft,
        unresolved_issues=[],
    )


def _extract_json(text: str) -> str:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1) if fenced else text


def parse_model_response(raw: str | dict[str, Any] | RefinerOutput) -> RefinerOutput:
    if isinstance(raw, RefinerOutput):
        return raw
    try:
        if isinstance(raw, str):
            raw = json.loads(_extract_json(raw))
        return RefinerOutput.model_validate(raw)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise RefinerResponseError(f"invalid structured model response: {exc}") from exc


def validate_output(case: RefinerInput, output: RefinerOutput) -> RefinerOutput:
    """Enforce properties that can be proven without another clinical judge."""

    flags = case.verifier_report.flagged_issues
    if not flags:
        if output != unchanged_output(case):
            raise RefinerResponseError("a no-flag case must preserve the draft exactly")
        return output

    if not output.needs_refinement and output.refined_text != case.draft:
        raise RefinerResponseError("needs_refinement=false requires exact draft preservation")
    if output.needs_refinement and output.refined_text == case.draft and not output.unresolved_issues:
        raise RefinerResponseError(
            "needs_refinement=true requires a correction or an unresolved flagged issue"
        )
    unknown = [issue for issue in output.unresolved_issues if issue not in flags]
    if unknown:
        raise RefinerResponseError(
            "unresolved_issues must copy supplied flags exactly; unknown values: " + repr(unknown)
        )
    return output

