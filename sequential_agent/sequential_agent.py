
import asyncio
import os
import sys
import json
import csv
import inspect
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from google.adk.agents import SequentialAgent, LoopAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# --------------------------------------------------
# PROJECT SETUP
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "sequential_agent"))

load_dotenv(ROOT / ".env")

from shared_test_cases import load_test_cases
from eval_harness import readability_scores

# --------------------------------------------------
# IMPORT AGENTS
# --------------------------------------------------

from extractor.agent import build_extractor_agent
from extractor.preprocess import preprocess
from simplifier_agent import simplifier_agent
from verifier import create_verifier_agent
from refiner import RefinerConfig, create_pipeline_refiner_agent
from readability_agent import readability_agent

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Evaluate all 40 cases
LIMIT = 20

CASE_DELAY_SECONDS = 5

OUTPUT_DIR = ROOT / "evaluation" / "results"

JSON_OUTPUT = OUTPUT_DIR / "pipeline_results.json"
CSV_OUTPUT = OUTPUT_DIR / "pipeline_metrics.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# EXTRACTOR TO SIMPLIFIER ADAPTER
# --------------------------------------------------

def adapt_atoms(state):
    extractor_output = state.get("atoms", {})

    if isinstance(extractor_output, dict):
        extractor_output = extractor_output.get("atoms", [])

    if not isinstance(extractor_output, list):
        return []

    adapted_atoms = []

    for atom in extractor_output:
        if not isinstance(atom, dict):
            continue

        adapted_atoms.append({
            "text": atom.get("claim", atom.get("text", "")),
            "type": atom.get("type", "")
        })

    return adapted_atoms


original_simplifier_callback = simplifier_agent.before_model_callback


async def simplifier_adapter(callback_context, llm_request):
    original_atoms = callback_context.state.get("atoms")

    callback_context.state["atoms"] = adapt_atoms(
        callback_context.state
    )

    try:
        result = original_simplifier_callback(
            callback_context,
            llm_request
        )

        if inspect.isawaitable(result):
            return await result

        return result

    finally:
        callback_context.state["atoms"] = original_atoms


simplifier_agent.before_model_callback = simplifier_adapter

# --------------------------------------------------
# BUILD AGENTS
# --------------------------------------------------

extractor_agent = build_extractor_agent(model=MODEL)

verifier_agent = create_verifier_agent(model=MODEL)

refiner_agent = create_pipeline_refiner_agent(
    RefinerConfig(model=MODEL)
)

refine_loop = LoopAgent(
    name="RefineLoop",
    sub_agents=[
        verifier_agent,
        refiner_agent,
    ],
    max_iterations=2,
)

root_agent = SequentialAgent(
    name="FaithfulMedPipeline",
    sub_agents=[
        extractor_agent,
        simplifier_agent,
        refine_loop,
        readability_agent,
    ],
)

# --------------------------------------------------
# RESULT HELPERS
# --------------------------------------------------

def get_final_text(value):

    if isinstance(value, str):
        return value.strip() or None

    if isinstance(value, dict):
        for key in (
            "text",
            "final_answer",
            "answer",
            "output",
            "explanation",
            "draft",
        ):
            candidate = value.get(key)

            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

    return None


def calculate_final_metrics(final_answer):

    if not final_answer:
        return None

    try:
        return readability_scores(final_answer)

    except Exception as error:
        print(f"[WARNING] Readability calculation failed: {error}")
        return None


# --------------------------------------------------
# RUN ONE CASE
# --------------------------------------------------

async def main(source_text, case_id):

    session_service = InMemorySessionService()

    session_id = f"test_session_{case_id}"

    runner = Runner(
        agent=root_agent,
        app_name="faithfulmed",
        session_service=session_service,
    )

    await session_service.create_session(
        app_name="faithfulmed",
        user_id="test_user",
        session_id=session_id,
        state={"source_text": source_text},
    )

    message = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=preprocess(source_text)
            )
        ],
    )

    print("\nStarting FaithfulMed pipeline...\n")

    async for event in runner.run_async(
        user_id="test_user",
        session_id=session_id,
        new_message=message,
    ):

        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"\n{event.author}:\n{part.text}")

    session = await session_service.get_session(
        app_name="faithfulmed",
        user_id="test_user",
        session_id=session_id,
    )

    case_results = {
        "case_id": case_id,
        "atoms": session.state.get("atoms"),
        "simplified_atoms": session.state.get("simplified_atoms"),
        "draft": session.state.get("draft"),
        "verdict": session.state.get("verdict"),
        "final": session.state.get("final"),
        "readability_scores": session.state.get("readability_scores"),
        "readability_passed": session.state.get("readability_passed"),
    }

    final_answer = get_final_text(
        case_results.get("final")
    )

    case_results["final_answer"] = final_answer

    case_results["readability_metrics"] = (
        calculate_final_metrics(final_answer)
    )

    case_results["final_word_count"] = (
        len(final_answer.split()) if final_answer else None
    )

    case_results["status"] = (
        "success" if final_answer else "missing_final_answer"
    )

    print("\n===== FINAL RESULTS =====")
    print(f"Case ID: {case_id}")
    print(f"Final answer available: {bool(final_answer)}")
    print(f"Readability metrics: {case_results['readability_metrics']}")

    return case_results


# --------------------------------------------------
# SAVE JSON: INDIVIDUAL CASE RESULTS
# --------------------------------------------------

def save_json(results):

    with open(JSON_OUTPUT, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


# --------------------------------------------------
# CALCULATE DATASET AVERAGES
# --------------------------------------------------

def calculate_dataset_averages(results):

    metric_keys = {
        "flesch_kincaid_grade": "average_flesch_kincaid",
        "smog_index": "average_smog",
        "medical_term_density": "average_medical_term_density",
        "word_count": "average_word_count",
    }

    averages = {}

    for source_key, output_key in metric_keys.items():

        values = []

        for result in results:

            metrics = result.get("readability_metrics") or {}

            value = metrics.get(source_key)

            if isinstance(value, (int, float)):
                values.append(value)

        averages[output_key] = (
            round(mean(values), 4) if values else None
        )

    return averages


# --------------------------------------------------
# SAVE CSV: ONE DATASET SUMMARY ROW
# --------------------------------------------------

def save_csv(results):

    successful = sum(
        result.get("status") == "success"
        for result in results
    )

    failed = len(results) - successful

    averages = calculate_dataset_averages(results)

    fields = [
        "total_cases",
        "successful_cases",
        "failed_cases",
        "average_flesch_kincaid",
        "average_smog",
        "average_medical_term_density",
        "average_word_count",
    ]

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerow({
            "total_cases": len(results),
            "successful_cases": successful,
            "failed_cases": failed,
            **averages,
        })


# --------------------------------------------------
# RUN DATASET
# --------------------------------------------------

async def run_all_cases():

    cases = load_test_cases()[:LIMIT]

    print("\n======================================")
    print("FAITHFULMED SEQUENTIAL AGENT EVALUATION")
    print("======================================")

    print(f"Model: {MODEL}")
    print(f"Cases loaded: {len(cases)}")
    print(f"JSON output: {JSON_OUTPUT}")
    print(f"CSV output: {CSV_OUTPUT}")

    all_results = []

    for index, case in enumerate(cases, start=1):

        case_id = str(case["case_id"])

        print(
            f"\n========== CASE {index}/{len(cases)}: "
            f"{case_id} =========="
        )

        try:

            result = await main(
                source_text=case["source_text"],
                case_id=case_id,
            )

            result["question"] = case["question"]
            result["reference_answer"] = case["reference_answer"]
            result["nuggets"] = case["nuggets"]

        except Exception as error:

            print(f"\n[FAILED] Case {case_id}: {error}")

            result = {
                "case_id": case_id,
                "question": case["question"],
                "reference_answer": case["reference_answer"],
                "nuggets": case["nuggets"],
                "status": "failed",
                "error": str(error),
                "final_answer": None,
                "readability_metrics": None,
            }

        all_results.append(result)

        # Save progress after every case
        save_json(all_results)
        save_csv(all_results)

        print(
            f"\n[PROGRESS] Saved {len(all_results)} "
            f"of {len(cases)} cases."
        )

        if index < len(cases):
            await asyncio.sleep(CASE_DELAY_SECONDS)

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    successful = sum(
        result.get("status") == "success"
        for result in all_results
    )

    failed = len(all_results) - successful

    averages = calculate_dataset_averages(all_results)

    print("\n======================================")
    print("EVALUATION COMPLETE")
    print("======================================")

    print(f"Total cases: {len(cases)}")
    print(f"Successful: {successful}")
    print(f"Failed or missing final answer: {failed}")

    print("\nDATASET AVERAGE METRICS")

    for metric, value in averages.items():
        print(f"{metric}: {value}")

    print(f"\nJSON: {JSON_OUTPUT}")
    print(f"CSV: {CSV_OUTPUT}")


if __name__ == "__main__":
    asyncio.run(run_all_cases())
