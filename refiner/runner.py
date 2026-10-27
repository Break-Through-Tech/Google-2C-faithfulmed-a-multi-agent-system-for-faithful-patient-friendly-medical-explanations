"""Standalone execution layer for one Refiner invocation."""

from __future__ import annotations

import asyncio
import inspect
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agent import RefinerConfig, create_refiner_agent
from .prompts import build_case_message
from .schemas import RefinerInput, RefinerOutput, VerifierReport
from .validation import (
    RefinerConfigurationError,
    RefinerResponseError,
    parse_model_response,
    unchanged_output,
    validate_output,
)

ModelInvoker = Callable[[RefinerInput], Any | Awaitable[Any]]


class RefinerRunner:
    """Runs ADK in production and accepts an injected invoker for offline tests."""

    def __init__(
        self,
        config: RefinerConfig | None = None,
        model_invoker: ModelInvoker | None = None,
    ) -> None:
        self.config = config or RefinerConfig.from_env()
        self.model_invoker = model_invoker
        self.last_latency_seconds: float | None = None
        self.last_usage_metadata: dict[str, Any] | None = None

    async def refine_case(self, case: RefinerInput) -> RefinerOutput:
        started = time.perf_counter()
        self.last_usage_metadata = None
        try:
            # Hard boundary: neither faithful=false nor readability=false authorizes a model call.
            if not case.verifier_report.flagged_issues:
                return unchanged_output(case)
            if self.model_invoker is not None:
                raw = self.model_invoker(case)
                if inspect.isawaitable(raw):
                    raw = await self._await_model(raw)
            else:
                raw = await self._await_model(self._invoke_adk(case))
            return validate_output(case, parse_model_response(raw))
        finally:
            self.last_latency_seconds = time.perf_counter() - started

    async def _await_model(self, awaitable: Awaitable[Any]) -> Any:
        """Apply one end-to-end deadline across the live call and ADK retries."""

        timeout = self.config.request_timeout_seconds
        if timeout <= 0:
            raise RefinerConfigurationError(
                "FAITHFULMED_REFINER_TIMEOUT_SECONDS must be greater than zero"
            )
        try:
            return await asyncio.wait_for(awaitable, timeout=timeout)
        except TimeoutError as exc:
            raise RefinerResponseError(
                f"Gemini did not complete within {timeout:g} seconds; "
                "the request was cancelled. Try again or choose another available model."
            ) from exc

    async def _invoke_adk(self, case: RefinerInput) -> str:
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise RefinerConfigurationError(
                "live refinement requires GEMINI_API_KEY (or GOOGLE_API_KEY); "
                "no-flag cases and injected/mock runs do not"
            )
        app_name = "faithfulmed_refiner"
        user_id = "standalone"
        session_id = f"refiner-{uuid.uuid4().hex}"
        sessions = InMemorySessionService()
        await sessions.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        runner = Runner(
            app_name=app_name,
            agent=create_refiner_agent(self.config),
            session_service=sessions,
        )
        message = types.Content(
            role="user",
            parts=[types.Part(text=build_case_message(case))],
        )
        final_text: str | None = None
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            usage = getattr(event, "usage_metadata", None)
            if usage is not None:
                if hasattr(usage, "model_dump"):
                    self.last_usage_metadata = usage.model_dump(exclude_none=True)
                else:
                    self.last_usage_metadata = {"value": str(usage)}
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(part.text or "" for part in event.content.parts)
        if not final_text:
            raise RefinerResponseError("ADK completed without a final text response")
        return final_text


async def refine_async(
    source_text: str,
    draft: str,
    verifier_report: dict[str, Any] | VerifierReport | None = None,
    *,
    example_id: str = "standalone",
    runner: RefinerRunner | None = None,
) -> RefinerOutput:
    case = RefinerInput(
        example_id=example_id,
        source_text=source_text,
        draft=draft,
        verifier_report=verifier_report or VerifierReport(),
    )
    return await (runner or RefinerRunner()).refine_case(case)


def refine(
    source_text: str,
    draft: str,
    verifier_report: dict[str, Any] | VerifierReport | None = None,
    *,
    example_id: str = "standalone",
    runner: RefinerRunner | None = None,
) -> RefinerOutput:
    """Perform one standalone verifier-directed refinement attempt."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            refine_async(
                source_text,
                draft,
                verifier_report,
                example_id=example_id,
                runner=runner,
            )
        )
    raise RuntimeError("refine() cannot run inside an event loop; await refine_async() instead")
