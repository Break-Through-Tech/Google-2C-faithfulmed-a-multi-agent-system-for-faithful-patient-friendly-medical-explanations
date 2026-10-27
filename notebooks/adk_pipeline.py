"""
FaithfulMed — minimal Google ADK pipeline skeleton.

Shows the *shape* the SME review asked for: a deterministic DAG (SequentialAgent), not open-ended
recursion, with a single BOUNDED loop (LoopAgent, max_iterations=2) for Refiner <-> Verifier.

This is a scaffold to react to, not a finished pipeline. Each fellow fleshes out their own agent's
instruction, output schema, and tools.

Setup:
    pip install google-adk
    export GEMINI_API_KEY=...        # free key from https://aistudio.google.com
Docs: https://google.github.io/adk-docs

Note: ADK's exact class/kwarg names evolve — check the docs for your installed version. The intent
(SequentialAgent for the DAG, LoopAgent with a hard iteration cap for the refine step) is the point.
"""
from pathlib import Path
import sys

# Allow both ``python notebooks/adk_pipeline.py`` and module-style execution from the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent
from readability_agent import readability_agent

from refiner import RefinerConfig, create_pipeline_refiner_agent

MODEL = "gemini-3.5-flash-lite"  # stable free-tier-friendly fallback; swap for the leaderboard

# 1) Extractor — pull clinical "atoms" (facts + types) from the source document.
extractor = LlmAgent(
    name="Extractor",
    model=MODEL,
    instruction=(
        "Extract every clinically meaningful fact from the SOURCE as a JSON list of atoms, each "
        "with {text, type} where type is one of: diagnosis, medication, lab_value, procedure, "
        "instruction. Do not add facts that are not in the SOURCE."
    ),
    output_key="atoms",
)

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

# 4) Refiner — typed standalone implementation adapted to pipeline state.
# The Runner/session that starts this pipeline must seed state["source_text"] with the original
# document. The adapter reads source_text + draft + verdict and overwrites draft with plain text.
refiner = create_pipeline_refiner_agent(RefinerConfig(model=MODEL))

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
