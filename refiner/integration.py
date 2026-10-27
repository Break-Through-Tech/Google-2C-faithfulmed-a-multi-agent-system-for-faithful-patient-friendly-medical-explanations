"""ADK state adapter for the integrated FaithfulMed pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.workflow import RetryConfig
from google.genai import types
from pydantic import ValidationError

from .agent import RefinerConfig
from .prompts import PIPELINE_SYSTEM_INSTRUCTION, build_case_message
from .schemas import RefinerInput, VerifierReport
from .validation import RefinerConfigurationError

PIPELINE_SOURCE_KEY = "source_text"
PIPELINE_DRAFT_KEY = "draft"
PIPELINE_VERDICT_KEY = "verdict"


def _parse_verdict(value: Any) -> VerifierReport:
    if isinstance(value, VerifierReport):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RefinerConfigurationError(
                "pipeline state 'verdict' must be a VerifierReport or JSON object"
            ) from exc
    try:
        return VerifierReport.model_validate(value)
    except ValidationError as exc:
        raise RefinerConfigurationError(
            "pipeline state 'verdict' does not match the Verifier contract"
        ) from exc


def pipeline_case_from_state(state: Mapping[str, Any]) -> RefinerInput:
    """Translate shared pipeline state into the standalone typed contract."""

    missing = [
        key
        for key in (PIPELINE_SOURCE_KEY, PIPELINE_DRAFT_KEY, PIPELINE_VERDICT_KEY)
        if key not in state
    ]
    if missing:
        raise RefinerConfigurationError(
            "integrated Refiner requires pipeline state keys: "
            + ", ".join((PIPELINE_SOURCE_KEY, PIPELINE_DRAFT_KEY, PIPELINE_VERDICT_KEY))
            + f"; missing: {', '.join(missing)}"
        )
    try:
        return RefinerInput(
            example_id=str(state.get("example_id", "pipeline")),
            source_text=state[PIPELINE_SOURCE_KEY],
            draft=state[PIPELINE_DRAFT_KEY],
            verifier_report=_parse_verdict(state[PIPELINE_VERDICT_KEY]),
        )
    except ValidationError as exc:
        raise RefinerConfigurationError("invalid integrated Refiner pipeline state") from exc


def build_pipeline_instruction(context: ReadonlyContext) -> str:
    """Build one instruction from the latest Verifier and draft state."""

    case = pipeline_case_from_state(context.state)
    return PIPELINE_SYSTEM_INSTRUCTION + "\n\n" + build_case_message(case)


def create_pipeline_refiner_agent(config: RefinerConfig | None = None) -> LlmAgent:
    """Create the drop-in Refiner node for the repository's bounded ADK loop.

    Input state: ``source_text`` (str), ``draft`` (str), ``verdict`` (Verifier report).
    Output state: ``draft`` (str), intentionally overwriting the current draft so the next
    Verifier iteration and the Readability agent receive the corrected explanation.
    """

    config = config or RefinerConfig.from_env()
    return LlmAgent(
        name="Refiner",
        description="Applies only verifier-flagged, source-supported corrections.",
        model=config.model,
        instruction=build_pipeline_instruction,
        output_key=PIPELINE_DRAFT_KEY,
        retry_config=RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=4.0,
            backoff_factor=2.0,
            jitter=0.25,
            exceptions=["ServerError"],
        ),
        generate_content_config=types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )
