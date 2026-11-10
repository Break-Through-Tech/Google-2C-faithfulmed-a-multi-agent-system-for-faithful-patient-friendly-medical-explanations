"""
ADK wiring for the Extractor.

ADK is imported inside functions so the rest of the package (schema, checks, eval) works
without google-adk installed.

The API key is read from .env (GOOGLE_API_KEY) by the google-genai client; it is never
passed or printed here.

Post-processing (ids, evidence check, hedge check) runs in an after_agent_callback, so it
happens identically whether the Extractor runs alone (extract()) or inside the team's
SequentialAgent. Downstream agents read session state["atoms"], a CheckedAtoms dict.
Callers should send preprocess(text) as the user message.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from extractor.checks import assign_ids, check_evidence, check_hedges
from extractor.preprocess import preprocess
from extractor.prompt import EXTRACTOR_PROMPT, PROMPT_VERSION
from extractor.schema import CheckedAtoms, ExtractionResult, ExtractorOutput, Provenance

log = logging.getLogger("faithfulmed.extractor")

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
# Same switch and default as notebooks/readability_agent.py, so one env var sets the model
# for the whole pipeline. The served model_version is still recorded per call.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_SEED = 42
TEMPERATURE = 0.0
APP_NAME = "faithfulmed-extractor"
# google-genai defaults to 5 attempts per request. The API quota is shared with the team,
# so cap it: one retry, after a 5 s backoff, for transient errors (429/5xx) only.
MAX_ATTEMPTS = 2
RETRY_INITIAL_DELAY_S = 5.0
STATE_KEY = "atoms"  # read by the Simplifier/Verifier as {atoms}
_MODEL_VERSION_KEY = "extractor_model_version"  # kept in state as provenance


def _log_model_call(callback_context, llm_response):
    """after_model_callback: log model name + served version for every call."""
    version = getattr(llm_response, "model_version", None)
    log.info("extractor call agent=%s model_version=%s", callback_context.agent_name, version)
    callback_context.state[_MODEL_VERSION_KEY] = version
    return None  # keep the response unchanged


def _user_text(content) -> str:
    if content is None or not content.parts:
        return ""
    return "".join(p.text or "" for p in content.parts)


def postprocess_atoms(callback_context):
    """after_agent_callback: replace raw LLM atoms in state with CheckedAtoms.

    Fails loudly if the model output is missing or invalid (e.g. truncated JSON on long
    sources) rather than passing an incomplete atom list downstream.
    """
    raw = callback_context.state.get(STATE_KEY)
    if raw is None:
        raise RuntimeError("Extractor produced no output in state")
    if isinstance(raw, str):
        drafts = ExtractorOutput.model_validate_json(raw).atoms
    else:
        drafts = ExtractorOutput.model_validate(raw).atoms

    source = preprocess(_user_text(callback_context.user_content))
    atoms = assign_ids(drafts)
    checked = CheckedAtoms(
        atoms=atoms,
        evidence_failures=check_evidence(atoms, source),
        hedge_flags=check_hedges(atoms),
        prompt_version=PROMPT_VERSION,
        model_version=callback_context.state.get(_MODEL_VERSION_KEY),
    )
    callback_context.state[STATE_KEY] = checked.model_dump()
    return None  # keep the agent's events unchanged


def build_extractor_agent(model: str = DEFAULT_MODEL, seed: int = DEFAULT_SEED):
    from google.adk.agents import LlmAgent
    from google.adk.models import Gemini
    from google.genai import types

    return LlmAgent(
        name="Extractor",
        model=Gemini(
            model=model,
            retry_options=types.HttpRetryOptions(
                attempts=MAX_ATTEMPTS, initial_delay=RETRY_INITIAL_DELAY_S
            ),
        ),
        # A callable instruction skips ADK's {state_var} templating, so braces in the
        # prompt can never be mistaken for state lookups.
        instruction=lambda _ctx: EXTRACTOR_PROMPT,
        output_schema=ExtractorOutput,
        output_key=STATE_KEY,
        generate_content_config=types.GenerateContentConfig(temperature=TEMPERATURE, seed=seed),
        after_model_callback=_log_model_call,
        after_agent_callback=postprocess_atoms,
    )


async def _run_once(agent, text: str) -> dict:
    """Send one source text through the agent; return the final state["atoms"]."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id="eval")
    message = types.Content(role="user", parts=[types.Part(text=text)])
    async for _ in runner.run_async(user_id="eval", session_id=session.id, new_message=message):
        pass
    session = await runner.session_service.get_session(
        app_name=APP_NAME, user_id="eval", session_id=session.id
    )
    return session.state[STATE_KEY]


def extract(source_id: str, text: str, model: str = DEFAULT_MODEL,
            seed: int = DEFAULT_SEED) -> ExtractionResult:
    """Run the Extractor on ONE source text. Makes one API call."""
    agent = build_extractor_agent(model=model, seed=seed)
    checked = CheckedAtoms.model_validate(asyncio.run(_run_once(agent, preprocess(text))))
    return ExtractionResult(
        source_id=source_id,
        atoms=checked.atoms,
        evidence_failures=checked.evidence_failures,
        hedge_flags=checked.hedge_flags,
        provenance=Provenance(
            model=model,
            model_version=checked.model_version,
            temperature=TEMPERATURE,
            seed=seed,
            prompt_version=PROMPT_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )
