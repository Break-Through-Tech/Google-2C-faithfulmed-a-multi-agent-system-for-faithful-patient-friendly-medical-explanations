"""Verifier agent: checks a draft explanation against its source facts.

Reads from session state:  `atoms` (the extracted facts) and `draft` (the text to check).
Writes to state["verdict"]: {"faithful", "unsupported_claims", "omissions", "reading_level_ok"}.

Usage in the pipeline:
    from verifier import create_verifier_agent
    verifier = create_verifier_agent(MODEL)

Model and API key (this file has no settings of its own):
- Model: pass the pipeline's MODEL to create_verifier_agent. If you pass nothing, it uses
  VERIFIER_DEFAULT_MODEL below.
- Key: Gemini models read the key from the environment, once for the whole pipeline, the same
  as the other agents. Set GOOGLE_API_KEY before running, for example
      export GOOGLE_API_KEY="your-key"            (macOS / Linux)
      $env:GOOGLE_API_KEY="your-key"              (Windows PowerShell)
  or in a notebook:  os.environ["GOOGLE_API_KEY"] = getpass.getpass()
  Never put a key in this file or in the repo.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from pydantic import BaseModel

# Used when no model is passed in. The pipeline normally passes its own MODEL.
VERIFIER_DEFAULT_MODEL = "gemini-3.8-flash"


class Verdict(BaseModel):
    """The output schema. ADK makes the model return exactly these four fields."""
    unsupported_claims: list[str]
    omissions: list[str]
    reading_level_ok: bool
    faithful: bool


VERDICT_KEYS = ["faithful", "unsupported_claims", "omissions", "reading_level_ok"]


# ----------------------------------------------------------------------------
# Prompt. RULES is shared by both modes and is the single copy of the rules.
# If you change RULES in a notebook, paste the new version here (and nowhere else).
# ----------------------------------------------------------------------------
INTRO_MEDAESQA = """You are a clinical faithfulness verifier. Judge the TEXT against the SOURCE only.

The SOURCE is a set of short evidence excerpts from the papers the TEXT cites.

"""

INTRO_PIPELINE = """You are a clinical faithfulness verifier. Judge the TEXT against the SOURCE only.

The SOURCE is a list of clinical atoms (facts) extracted from the original medical report.
The TEXT is a patient-friendly draft explaining it.

"""

RULES = """Work through these steps silently, then return only the JSON object.

1. Use only the SOURCE. A statement that is true in general medicine but is not in the SOURCE
   is not supported. Do not assume the SOURCE is complete: a claim it does not state is "not
   stated", which is different from "wrong".

2. Split the TEXT into claims. A clinical claim tells the reader something about a diagnosis,
   test result, number, treatment, drug, dose, cause, effect, risk, or what to do. Ignore
   framing words ("Additionally", "It is important to note"), generic reassurance, advice such
   as "ask your doctor", and plain explanations of a term that add no new clinical fact.

3. Check every clinical claim against the SOURCE:
   supported = the SOURCE states it, entails it, or says the same thing in different words.
               Hedging must match: a SOURCE that says "may" or reports mixed results supports
               only a hedged claim.
   contradicted = the SOURCE says something different or opposite: the wrong direction
               (increase vs decrease), a wrong number, unit, dose, frequency or duration, an
               effect that continues where the SOURCE says it stops, a definite statement where
               the SOURCE is uncertain, an effect where the SOURCE found none, found it not
               significant, or found it only for something else, or the opposite of a negative
               finding ("does not", "no effect").
   not stated = the SOURCE says nothing about it.
   Make sure the SOURCE is about the same thing the claim is about: the same condition, drug,
   test, side of the body and population, and the same relationship between them. Do not carry
   a fact over from a similar thing (one drug to another, type 1 to type 2 diabetes, adults to
   children). Sharing a topic is not support. Words like causes, prevents, cures, reduces risk,
   is safe, does not cause, or is not needed need an explicit statement in the SOURCE. Do not
   decide that a value is normal or abnormal from the number alone; use what the SOURCE says.

4. unsupported_claims: one entry for every clinical claim in the TEXT that is contradicted or
   not stated. Each entry is one short sentence that names the specific detail so it can be
   found in the TEXT, says whether the SOURCE contradicts it or does not state it, and ends
   with a few words on the fix (remove it, correct it to the SOURCE value, or soften it).
   Style:
   "The dose is incorrect: the SOURCE says 500 mg twice daily, so correct it."
   "The claim that the drug improves survival is not stated in the SOURCE, so remove it."
   Only list problems that really are in the TEXT.

5. omissions: list a fact only if all three are true: the SOURCE states it, the TEXT does not
   already convey it (even in different words), and leaving it out would change what the
   patient understands or does next (a main diagnosis, a medication change, a result the SOURCE
   calls abnormal or critical, follow-up, a warning). Do not list facts just for completeness.
   Each entry is one short sentence. Return an empty list if nothing important is missing.

6. reading_level_ok: true if the wording fits roughly an 8th-grade reading level.

7. faithful: true only if unsupported_claims is empty.

Return exactly one JSON object with exactly these four keys and no others, and no text outside it:
{"faithful": true or false, "unsupported_claims": ["..."], "omissions": ["..."], "reading_level_ok": true or false}"""


def format_atoms(atoms: Any) -> str:
    """Accept atoms as plain text, a JSON string, or an already-parsed list/dict."""
    if isinstance(atoms, str):
        try:
            atoms = json.loads(atoms)
        except (json.JSONDecodeError, TypeError):
            return atoms
        if isinstance(atoms, str):
            return atoms
    return json.dumps(atoms, ensure_ascii=False, indent=2)


def build_prompt(source: str, draft: str, mode: str = "pipeline") -> str:
    """mode "medaesqa" is the calibration setting (short evidence excerpts as the source);
    mode "pipeline" is for the Extractor's atoms."""
    intro = INTRO_MEDAESQA if mode == "medaesqa" else INTRO_PIPELINE
    return intro + "SOURCE:\n" + source + "\n\nTEXT TO AUDIT:\n" + draft + "\n\n" + RULES


def _instruction_from_state(ctx) -> str:
    """ADK calls this before each model call. Building the prompt in code means braces in
    the text are never mistaken for state placeholders."""
    state = ctx.state
    return build_prompt(
        format_atoms(state.get("atoms", "")),
        str(state.get("draft", "")),
        mode=state.get("verifier_mode", "pipeline"),
    )


def create_verifier_agent(model: str = VERIFIER_DEFAULT_MODEL) -> Any:
    try:
        from google.adk.agents import LlmAgent
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("google-adk is not installed. Run: pip install -r requirements.txt") from exc
    return LlmAgent(
        name="Verifier",
        model=model,
        description="Checks a draft explanation against its source facts.",
        instruction=_instruction_from_state,
        output_schema=Verdict,
        output_key="verdict",
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
    )


def parse_verdict(text: str) -> dict:
    """Accept a reply only if it is exactly the four-key object; otherwise raise ValueError."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    d = json.loads(text)
    if not isinstance(d, dict) or set(d) != set(VERDICT_KEYS):
        raise ValueError(f"wrong keys: {sorted(d) if isinstance(d, dict) else type(d).__name__}")
    if not isinstance(d["faithful"], bool) or not isinstance(d["reading_level_ok"], bool):
        raise ValueError("faithful and reading_level_ok must be true/false")
    for k in ("unsupported_claims", "omissions"):
        if not isinstance(d[k], list) or not all(isinstance(x, str) for x in d[k]):
            raise ValueError(f"{k} must be a list of strings")
    return {k: d[k] for k in VERDICT_KEYS}


async def _run_agent(agent, state: dict, message: str,
                     on_text: Optional[Callable[[str, str], None]] = None) -> str:
    """Run one agent once on its own, outside the pipeline. Returns the final reply text.
    on_text(author, text) is called for every event that carries text."""
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    sessions = InMemorySessionService()
    runner = Runner(agent=agent, app_name="verifier", session_service=sessions)
    session = await sessions.create_session(app_name="verifier", user_id="verifier", state=state)
    content = types.Content(role="user", parts=[types.Part(text=message)])

    final = ""
    async for event in runner.run_async(user_id="verifier", session_id=session.id, new_message=content):
        parts = event.content.parts if event.content and event.content.parts else []
        text = "".join(p.text or "" for p in parts)
        if text and on_text:
            on_text(event.author, text)
        if text and event.is_final_response():
            final = text
    return final


async def _verify(source: str, draft: str, mode: str, model: str, on_text=None) -> dict:
    agent = create_verifier_agent(model)
    state = {"atoms": source, "draft": draft, "verifier_mode": mode}
    raw = await _run_agent(agent, state, "Check the draft against the source.", on_text)
    return parse_verdict(raw)


async def run_verifier(source_text: str, draft: str, model: str = VERIFIER_DEFAULT_MODEL) -> dict:
    """Ad-hoc check of a draft against plain source text."""
    return await _verify(source_text, draft, "pipeline", model)


async def run_medaesqa_verifier(evidence_text: str, answer_sentence: str,
                                model: str = VERIFIER_DEFAULT_MODEL) -> dict:
    """The calibration setting: one MedAESQA sentence against its evidence excerpts."""
    return await _verify(evidence_text, answer_sentence, "medaesqa", model)


async def run_production_verifier(atoms: Any, draft: str, model: str = VERIFIER_DEFAULT_MODEL,
                                  *, on_text: Optional[Callable[[str, str], None]] = None) -> dict:
    """The pipeline setting: the Extractor's atoms against the draft. Pass on_text to see replies as they arrive."""
    return await _verify(format_atoms(atoms), draft, "pipeline", model, on_text)
