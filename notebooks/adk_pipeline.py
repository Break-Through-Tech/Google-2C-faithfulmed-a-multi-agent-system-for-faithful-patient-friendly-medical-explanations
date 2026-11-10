"""
FaithfulMed — minimal Google ADK pipeline skeleton.

Shows the *shape* the SME review asked for: a deterministic DAG (SequentialAgent), not open-ended
recursion, with a single BOUNDED loop (LoopAgent, max_iterations=2) for Refiner <-> Verifier.

This is a scaffold to react to, not a finished pipeline. Each fellow fleshes out their own agent's
instruction, output schema, and tools.

Setup:
    pip install google-adk
    GOOGLE_API_KEY=... in .env        # free key from https://aistudio.google.com
    GEMINI_MODEL=...  (optional)      # defaults to gemini-3.5-flash-lite
Docs: https://google.github.io/adk-docs

Note: ADK's exact class/kwarg names evolve — check the docs for your installed version. The intent
(SequentialAgent for the DAG, LoopAgent with a hard iteration cap for the refine step) is the point.
"""
import os
import sys
from pathlib import Path

from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent
from readability_agent import readability_agent

# The extractor/ package lives at the repo root; make it importable from notebooks/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.agent import build_extractor_agent  # noqa: E402

# One switch for the whole pipeline, same as readability_agent.py. gemini-2.0-flash (the old
# default) is no longer served to new API keys.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# 1) Extractor — atomic, evidence-backed clinical facts (see extractor/). Writes state["atoms"]:
#    {atoms: [{id, claim, type, predicate, subject, object, hedge, evidence, importance}],
#     evidence_failures, hedge_flags, prompt_version, model_version}.
#    Send preprocess(text) from extractor.preprocess as the user message.
extractor = build_extractor_agent(model=MODEL)

# 2) Simplifier — rewrite into plain language, grounded by RAG (wire retrieval into instruction/tools).
simplifier = LlmAgent(
    name="Simplifier",
    model=MODEL,
    instruction=(
        "Rewrite the SOURCE for a patient at an 8th-grade reading level using only the extracted "
        "{atoms}. Use plain-language definitions retrieved from the glossary. Add nothing new."
    ),
    output_key="draft",
)

# 3) Verifier — LLM-as-judge: is every claim in {draft} supported by {atoms}? Flag omissions/additions.
verifier = LlmAgent(
    name="Verifier",
    model=MODEL,
    instruction=(
        "Compare {draft} against {atoms}. Return JSON: {faithful: bool, unsupported_claims: [...], "
        "omissions: [...], reading_level_ok: bool}. Be strict: an unsupported clinical claim is a failure."
    ),
    output_key="verdict",
)

# 4) Refiner — fix only what the Verifier flagged. No free rewriting.
refiner = LlmAgent(
    name="Refiner",
    model=MODEL,
    instruction=(
        "Given {draft} and {verdict}, produce a corrected draft that removes unsupported claims and "
        "restores omissions, changing nothing else. If {verdict.faithful} is true, return {draft} unchanged."
    ),
    output_key="draft",  # overwrites the draft so the loop re-verifies the corrected version
)

# Bounded refine loop: Verifier -> Refiner, at most twice. This is the ONLY loop in the system.
refine_loop = LoopAgent(
    name="RefineLoop",
    sub_agents=[verifier, refiner],
    max_iterations=2,
)


# The deterministic DAG: Extractor -> Simplifier -> (Verifier <-> Refiner, bounded) -> Readability.
faithfulmed = SequentialAgent(
    name="FaithfulMed",
    sub_agents=[extractor, simplifier, refine_loop, readability_agent],
)

if __name__ == "__main__":
    print("FaithfulMed ADK pipeline assembled:")
    print("  Extractor -> Simplifier -> [Verifier <-> Refiner x<=2] -> Readability")
    print("Run it with an ADK Runner + session (see the ADK quickstart) once GEMINI_API_KEY is set.")
