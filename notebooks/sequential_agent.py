
import asyncio
import os
import sys
from pathlib import Path

from google.adk.agents import SequentialAgent, LoopAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Add project directories to Python's import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import agents
from extractor.agent import build_extractor_agent
from extractor.preprocess import preprocess
from simplifier_agent import simplifier_agent
from verifier import create_verifier_agent
from refiner import RefinerConfig, create_pipeline_refiner_agent
from readability_agent import readability_agent


MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


# Convert Extractor output into the format Simplifier expects
def adapt_atoms(state):
    extractor_output = state.get("atoms", {})

    if isinstance(extractor_output, dict):
        extractor_output = extractor_output.get("atoms", [])

    return [
        {
            "text": atom.get("claim", atom.get("text", "")),
            "type": atom.get("type", "")
        }
        for atom in extractor_output
        if isinstance(atom, dict)
    ]


# Adapt Simplifier callback without modifying its original file
original_simplifier_callback = simplifier_agent.before_model_callback


async def simplifier_adapter(callback_context, llm_request):
    original_atoms = callback_context.state.get("atoms")

    callback_context.state["atoms"] = adapt_atoms(
        callback_context.state
    )

    try:
        return await original_simplifier_callback(
            callback_context, llm_request
        )
    finally:
        # Restore original Extractor output for downstream agents
        callback_context.state["atoms"] = original_atoms


simplifier_agent.before_model_callback = simplifier_adapter


# 1. Extractor
extractor_agent = build_extractor_agent(model=MODEL)


# 2. Verifier
verifier_agent = create_verifier_agent(model=MODEL)


# 3. Refiner
refiner_agent = create_pipeline_refiner_agent(
    RefinerConfig(model=MODEL)
)


# 4. Verifier-Refiner loop
refine_loop = LoopAgent(
    name="RefineLoop",
    sub_agents=[
        verifier_agent,
        refiner_agent,
    ],
    max_iterations=2,
)


# 5. Complete pipeline
root_agent = SequentialAgent(
    name="FaithfulMedPipeline",
    sub_agents=[
        extractor_agent,
        simplifier_agent,
        refine_loop,
        readability_agent,
    ],
)


async def main():
    session_service = InMemorySessionService()

    runner = Runner(
        agent=root_agent,
        app_name="faithfulmed",
        session_service=session_service,
    )

    source_text = """
    Patient presents with bilateral lower-extremity edema.
    Prescribed furosemide 20 mg daily.
    Follow up in 2 weeks.
    """

    await session_service.create_session(
        app_name="faithfulmed",
        user_id="test_user",
        session_id="test_session",
        state={
            "source_text": source_text,
        },
    )

    message = types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=preprocess(source_text))
        ],
    )

    print("\nStarting FaithfulMed pipeline...\n")

    async for event in runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"\n{event.author}:\n{part.text}")

    session = await session_service.get_session(
        app_name="faithfulmed",
        user_id="test_user",
        session_id="test_session",
    )

    print("\n===== FINAL RESULTS =====")
    print("Atoms:", session.state.get("atoms"))
    print("Simplified atoms:", session.state.get("simplified_atoms"))
    print("Draft:", session.state.get("draft"))
    print("Verdict:", session.state.get("verdict"))
    print("Final:", session.state.get("final"))
    print("Readability scores:", session.state.get("readability_scores"))
    print("Readability passed:", session.state.get("readability_passed"))


if __name__ == "__main__":
    asyncio.run(main())
