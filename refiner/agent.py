"""Google ADK agent factory for the FaithfulMed Refiner."""

from __future__ import annotations

import os
from dataclasses import dataclass

from google.adk.agents import LlmAgent
from google.adk.workflow import RetryConfig
from google.genai import types

from .prompts import PROMPT_VERSION, SYSTEM_INSTRUCTION

# The repository's original gemini-2.0-flash starter has been shut down.
# This stable Flash-Lite model is available to new projects and completed the
# project's five-case live structured-output run on 2026-10-26.
DEFAULT_MODEL = "gemini-3.5-flash-lite"


REFINER_OUTPUT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "needs_refinement": types.Schema(
            type=types.Type.BOOLEAN,
            description="Whether a valid correction or unresolved review issue exists.",
        ),
        "refined_text": types.Schema(
            type=types.Type.STRING,
            description="The minimally corrected draft, or the exact unchanged draft.",
        ),
        "unresolved_issues": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="Supplied Verifier flags that cannot be resolved from the source.",
        ),
    },
    required=["needs_refinement", "refined_text", "unresolved_issues"],
)


@dataclass(frozen=True)
class RefinerConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_output_tokens: int = 2048
    request_timeout_seconds: float = 45.0
    prompt_version: str = PROMPT_VERSION

    @classmethod
    def from_env(cls) -> "RefinerConfig":
        return cls(
            model=os.getenv("FAITHFULMED_REFINER_MODEL", DEFAULT_MODEL),
            temperature=float(os.getenv("FAITHFULMED_REFINER_TEMPERATURE", "0")),
            max_output_tokens=int(os.getenv("FAITHFULMED_REFINER_MAX_OUTPUT_TOKENS", "2048")),
            request_timeout_seconds=float(
                os.getenv("FAITHFULMED_REFINER_TIMEOUT_SECONDS", "45")
            ),
        )


def create_refiner_agent(config: RefinerConfig | None = None) -> LlmAgent:
    """Build one replaceable Gemini-backed ADK ``LlmAgent``."""

    config = config or RefinerConfig.from_env()
    return LlmAgent(
        name="UrologyRefiner",
        description="Applies only verifier-flagged, source-supported corrections.",
        model=config.model,
        instruction=SYSTEM_INSTRUCTION,
        # This explicit Gemini Schema avoids Pydantic's additionalProperties field while
        # allowing ADK to configure native structured output for this tool-free agent.
        output_schema=REFINER_OUTPUT_SCHEMA,
        output_key="refiner_result",
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
            # This agent has no callable tools. Explicitly disabling the GenAI SDK's default
            # AFC behavior removes its AsyncModels.generate_content recommendation warning.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )
