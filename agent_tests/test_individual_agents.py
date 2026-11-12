
import asyncio
import csv
import inspect
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

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
from extractor.agent import extract
from simplifier_agent import process_simplification
from verifier import run_verifier
from refiner.runner import refine_async
from readability_agent import process_readability
from eval_harness import readability_scores


MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

LIMIT = 25

CASE_DELAY_SECONDS = 10
AGENT_DELAY_SECONDS = 5

OUTPUT_FILE = ROOT / "agent_tests" / "individual_agent_metrics.csv"
SUMMARY_FILE = ROOT / "agent_tests" / "individual_agent_averages.csv"

client = genai.Client()

AGENTS = [
    "Extractor",
    "Simplifier",
    "Verifier",
    "Refiner",
    "Readability",
]

FIELDS = [
    "example_id",
    "model",
    "pipeline",
    "answer",
    "flesch_kincaid",
    "smog",
    "jargon_density",
    "word_count",
    "refusal",
    "faithfulness_score",
    "hallucination",
    "status",
    "error",
    "verifier_report",
    "agent_data",
]

# Holds one latest result per case, model, and agent.
results_by_key = {}


# --------------------------------------------------
# GENERAL HELPERS
# --------------------------------------------------

async def resolve(value):
    if inspect.isawaitable(value):
        return await value
    return value


def result_key(row):
    return (
        str(row["example_id"]),
        str(row["model"]),
        str(row["pipeline"]),
    )


def load_existing_results():
    """
    Load previous results so successful agents are not called again.
    Results from other models are ignored.
    """

    results_by_key.clear()

    if not OUTPUT_FILE.exists():
        return

    with open(
        OUTPUT_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row.get("model") != MODEL:
                continue

            if row.get("pipeline") not in AGENTS:
                continue

            key = result_key(row)

            results_by_key[key] = row

    print(
        f"[RESUME] Loaded "
        f"{len(results_by_key)} existing results for {MODEL}"
    )


def write_results():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    with open(
        temp_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(results_by_key.values())

    temp_file.replace(OUTPUT_FILE)


def save_row(row):
    """
    Insert or replace one result.
    Previously saved results for other cases and agents are preserved.
    """

    row.setdefault("agent_data", None)

    key = result_key(row)

    results_by_key[key] = row

    # Persist immediately after each agent.
    write_results()


def make_failure_row(example_id, pipeline, error):

    return {
        "example_id": str(example_id),
        "model": MODEL,
        "pipeline": pipeline,
        "answer": None,
        "flesch_kincaid": None,
        "smog": None,
        "jargon_density": None,
        "word_count": None,
        "refusal": None,
        "faithfulness_score": None,
        "hallucination": None,
        "status": "failed",
        "error": str(error),
        "verifier_report": None,
        "agent_data": None,
    }


def save_failure(example_id, pipeline, error):

    row = make_failure_row(
        example_id,
        pipeline,
        error,
    )

    save_row(row)

    return row


def save_skipped(example_id, pipeline, reason):

    row = make_failure_row(
        example_id,
        pipeline,
        reason,
    )

    row["status"] = "skipped"

    save_row(row)

    return row


def get_saved_result(example_id, pipeline):

    return results_by_key.get(
        (str(example_id), MODEL, pipeline)
    )


def already_succeeded(example_id, pipeline):

    row = get_saved_result(
        example_id,
        pipeline,
    )

    return (
        row is not None
        and row.get("status") == "success"
    )


def get_saved_answer(example_id, pipeline):

    row = get_saved_result(
        example_id,
        pipeline,
    )

    if not row or row.get("status") != "success":
        return None

    answer = row.get("answer")

    if not answer:
        return None

    return answer


def get_saved_agent_data(example_id, pipeline):

    row = get_saved_result(
        example_id,
        pipeline,
    )

    if not row:
        return None

    data = row.get("agent_data")

    if not data:
        return None

    try:
        return json.loads(data)
    except (TypeError, json.JSONDecodeError):
        return None


def get_saved_verifier_report(example_id):

    row = get_saved_result(
        example_id,
        "Verifier",
    )

    if not row or row.get("status") != "success":
        return None

    report = row.get("verifier_report")

    if not report:
        return None

    try:
        return json.loads(report)
    except (TypeError, json.JSONDecodeError):
        return None


def get_atom_text(atom):

    if isinstance(atom, dict):

        for key in ("text", "claim", "content", "statement"):

            if atom.get(key):
                return str(atom[key])

    for key in ("text", "claim", "content", "statement"):

        value = getattr(atom, key, None)

        if value:
            return str(value)

    return str(atom)


# --------------------------------------------------
# GENERATED TEXT EXTRACTION
# --------------------------------------------------

TEXT_KEYS = (
    "draft",
    "refined_text",
    "final_answer",
    "final_text",
    "simplified_text",
    "simplified_answer",
    "readable_text",
    "readability_text",
    "generated_text",
    "explanation",
    "answer",
    "output",
    "response",
    "content",
    "text",
)


def get_text(result):
    """
    Extract text from common output fields.

    Does not automatically treat arbitrary metadata values
    as generated answers.
    """

    if isinstance(result, str):

        text = result.strip()

        if text:
            return text

    if hasattr(result, "model_dump"):
        result = result.model_dump()

    elif not isinstance(result, (dict, list)):
        if hasattr(result, "__dict__"):
            result = vars(result)

    if isinstance(result, dict):

        # Search recognized fields first.
        for key in TEXT_KEYS:

            if key in result:

                try:
                    return get_text(result[key])
                except ValueError:
                    pass

        # Search nested dictionaries and lists.
        for value in result.values():

            if isinstance(value, (dict, list)) or hasattr(
                value, "model_dump"
            ):

                try:
                    return get_text(value)
                except ValueError:
                    pass

    elif isinstance(result, list):

        for item in result:

            if isinstance(item, str) and item.strip():
                return item.strip()

            if isinstance(item, (dict, list)) or hasattr(
                item, "model_dump"
            ):

                try:
                    return get_text(item)
                except ValueError:
                    pass

    raise ValueError(
        "Could not extract generated text. "
        f"Returned type: {type(result).__name__}. "
        f"Returned data: {str(result)[:1000]}"
    )


# --------------------------------------------------
# VERIFIER HELPERS
# --------------------------------------------------

def to_dict(value):

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "__dict__"):
        return vars(value)

    return {"result": str(value)}


def find_value(data, keys):

    if isinstance(data, dict):

        for key in keys:

            if key in data:
                return data[key]

        for value in data.values():

            found = find_value(value, keys)

            if found is not None:
                return found

    elif isinstance(data, list):

        for item in data:

            found = find_value(item, keys)

            if found is not None:
                return found

    return None


def normalize_score(value):

    if isinstance(value, (int, float)) and not isinstance(value, bool):

        score = float(value)

        if score > 1:
            score /= 100

        return max(0.0, min(1.0, score))

    return None


def extract_faithfulness_and_hallucination(report):

    report = to_dict(report)

    # Extract the Boolean returned by the Verifier.
    faithful = report.get("faithful")

    if isinstance(faithful, bool):
        faithfulness_score = 1.0 if faithful else 0.0
        hallucination = not faithful
    else:
        faithfulness_score = None
        hallucination = None

    return faithfulness_score, hallucination


# --------------------------------------------------
# READABILITY AND REFUSAL METRICS
# --------------------------------------------------

def is_refusal(text):

    patterns = [
        r"\bi cannot provide\b",
        r"\bi can't provide\b",
        r"\bi cannot answer\b",
        r"\bi can't answer\b",
        r"\bunable to answer\b",
        r"\bunable to provide\b",
        r"\bi cannot help with\b",
        r"\bi'm unable to\b",
    ]

    return any(
        re.search(pattern, text.lower())
        for pattern in patterns
    )


def build_answer_row(
    example_id,
    pipeline,
    answer,
):

    if not answer or not answer.strip():
        raise ValueError("Generated answer is empty.")

    readability = readability_scores(answer)

    return {
        "example_id": str(example_id),
        "model": MODEL,
        "pipeline": pipeline,
        "answer": answer,
        "flesch_kincaid": readability.get(
            "flesch_kincaid_grade"
        ),
        "smog": readability.get("smog_index"),
        "jargon_density": readability.get(
            "medical_term_density"
        ),
        "word_count": readability.get("word_count"),
        "refusal": is_refusal(answer),
        "faithfulness_score": None,
        "hallucination": None,
        "status": "success",
        "error": None,
        "verifier_report": None,
        "agent_data": None,
    }


# --------------------------------------------------
# INDIVIDUAL AGENT EVALUATION
# --------------------------------------------------

async def evaluate_case(case):

    example_id = str(case["case_id"])
    source_text = case["source_text"]

    completed = []

    print(f"\n========== CASE {example_id} ==========")

    # ----------------------------------------------
    # 1. EXTRACTOR
    # ----------------------------------------------

    if already_succeeded(example_id, "Extractor"):

        print("[RESUME] Extractor already completed.")

        extractor_answer = get_saved_answer(
            example_id,
            "Extractor",
        )

        saved_atoms = get_saved_agent_data(
            example_id,
            "Extractor",
        )

        if saved_atoms:
            atoms = saved_atoms

        else:
            # Compatibility with older CSV results.
            atoms = [
                {"text": line}
                for line in extractor_answer.splitlines()
                if line.strip()
            ]

    else:

        try:

            extraction = await asyncio.to_thread(
                extract,
                source_id=f"individual_test_{example_id}",
                text=source_text,
                model=MODEL,
            )

            atoms = [
                {"text": get_atom_text(atom)}
                for atom in extraction.atoms
            ]

            if not atoms:
                raise ValueError("Extractor returned no atoms.")

            extractor_answer = "\n".join(
                atom["text"] for atom in atoms
            )

            row = build_answer_row(
                example_id,
                "Extractor",
                extractor_answer,
            )

            row["agent_data"] = json.dumps(
                atoms,
                default=str,
            )

            save_row(row)
            completed.append(row)

            print(f"[SAVED] Extractor: {len(atoms)} atoms")

        except Exception as error:

            print(f"[FAILED] Extractor: {error}")

            failure = save_failure(
                example_id,
                "Extractor",
                error,
            )

            completed.append(failure)

            # Downstream agents cannot run without extraction.
            for agent in AGENTS[1:]:

                skipped = save_skipped(
                    example_id,
                    agent,
                    "Skipped because Extractor failed.",
                )

                completed.append(skipped)

            return completed

    await asyncio.sleep(AGENT_DELAY_SECONDS)

    # ----------------------------------------------
    # 2. SIMPLIFIER
    # ----------------------------------------------

    if already_succeeded(example_id, "Simplifier"):

        print("[RESUME] Simplifier already completed.")

        simplified_answer = get_saved_answer(
            example_id,
            "Simplifier",
        )

    else:

        try:

            simplifier_result = await resolve(
                process_simplification(
                    atoms=atoms,
                    client=client,
                    model=MODEL,
                )
            )

            simplified_answer = get_text(
                simplifier_result
            )

            row = build_answer_row(
                example_id,
                "Simplifier",
                simplified_answer,
            )

            save_row(row)
            completed.append(row)

            print("[SAVED] Simplifier")

        except Exception as error:

            print(f"[FAILED] Simplifier: {error}")

            failure = save_failure(
                example_id,
                "Simplifier",
                error,
            )

            completed.append(failure)

            # Verifier, Refiner, and Readability need Simplifier output.
            for agent in ("Verifier", "Refiner", "Readability"):

                skipped = save_skipped(
                    example_id,
                    agent,
                    "Skipped because Simplifier failed.",
                )

                completed.append(skipped)

            return completed

    await asyncio.sleep(AGENT_DELAY_SECONDS)

    # ----------------------------------------------
    # 3. VERIFIER
    # ----------------------------------------------

    if already_succeeded(example_id, "Verifier"):

        print("[RESUME] Verifier already completed.")

        verifier_report = get_saved_verifier_report(
            example_id
        )

    else:

        try:

            verifier_report = await resolve(
                run_verifier(
                    source_text=source_text,
                    draft=simplified_answer,
                    model=MODEL,
                )
            )

            verifier_data = to_dict(verifier_report)

            faithfulness, hallucination = extract_faithfulness_and_hallucination(
                verifier_report
            )
            report = verifier_data

            verifier_row = {
                "example_id": example_id,
                "model": MODEL,
                "pipeline": "Verifier",
                "answer": json.dumps(
                    verifier_data,
                    default=str,
                ),
                "flesch_kincaid": None,
                "smog": None,
                "jargon_density": None,
                "word_count": None,
                "refusal": None,
                "faithfulness_score": faithfulness,
                "hallucination": hallucination,
                "status": "success",
                "error": None,
                "verifier_report": json.dumps(
                    report,
                    default=str,
                ),
                "agent_data": None,
            }

            save_row(verifier_row)
            completed.append(verifier_row)

            print(
                "[SAVED] Verifier | "
                f"Faithfulness: {faithfulness} | "
                f"Hallucination: {hallucination}"
            )

        except Exception as error:

            print(f"[FAILED] Verifier: {error}")

            failure = save_failure(
                example_id,
                "Verifier",
                error,
            )

            completed.append(failure)

            verifier_report = None

    await asyncio.sleep(AGENT_DELAY_SECONDS)

    # ----------------------------------------------
    # 4. REFINER
    # ----------------------------------------------

    if already_succeeded(example_id, "Refiner"):

        print("[RESUME] Refiner already completed.")

        refined_answer = get_saved_answer(
            example_id,
            "Refiner",
        )

    else:

        try:

            refinement = await resolve(
                refine_async(
                    source_text=source_text,
                    draft=simplified_answer,
                    verifier_report=verifier_report,
                    example_id=example_id,
                )
            )

            refined_answer = get_text(refinement)

            row = build_answer_row(
                example_id,
                "Refiner",
                refined_answer,
            )

            save_row(row)
            completed.append(row)

            print("[SAVED] Refiner")

        except Exception as error:

            print(f"[FAILED] Refiner: {error}")

            failure = save_failure(
                example_id,
                "Refiner",
                error,
            )

            completed.append(failure)

            refined_answer = None

    await asyncio.sleep(AGENT_DELAY_SECONDS)

    # ----------------------------------------------
    # 5. READABILITY
    # ----------------------------------------------

    if already_succeeded(example_id, "Readability"):

        print("[RESUME] Readability already completed.")

    else:

        readability_input = (
            refined_answer
            if refined_answer is not None
            else simplified_answer
        )

        try:

            readability_result = await resolve(
                process_readability(
                    readability_input,
                    client=client,
                    model=MODEL,
                )
            )

            readability_answer = get_text(
                readability_result
            )

            row = build_answer_row(
                example_id,
                "Readability",
                readability_answer,
            )

            save_row(row)
            completed.append(row)

            print("[SAVED] Readability")

        except Exception as error:

            print(f"[FAILED] Readability: {error}")

            failure = save_failure(
                example_id,
                "Readability",
                error,
            )

            completed.append(failure)

    return completed


# --------------------------------------------------
# SUMMARY HELPERS
# --------------------------------------------------

def numeric_value(value):

    if value is None or value == "":
        return None

    try:
        number = float(value)

        if number != number:
            return None

        return number

    except (ValueError, TypeError):
        return None


def calculate_average(rows, metric):

    values = [
        numeric_value(row.get(metric))
        for row in rows
    ]

    values = [
        value for value in values
        if value is not None
    ]

    if not values:
        return None, 0

    return sum(values) / len(values), len(values)


def parse_bool(value):

    if isinstance(value, bool):
        return value

    if isinstance(value, str):

        normalized = value.strip().lower()

        if normalized in ("true", "1", "yes"):
            return True

        if normalized in ("false", "0", "no"):
            return False

    return None


def calculate_rate(rows, metric):

    values = [
        parse_bool(row.get(metric))
        for row in rows
    ]

    values = [
        value for value in values
        if value is not None
    ]

    if not values:
        return None, 0

    return sum(values) / len(values), len(values)


# --------------------------------------------------
# DATASET AVERAGES
# --------------------------------------------------

def save_summary(total_cases):

    metric_columns = [
        "flesch_kincaid",
        "smog",
        "jargon_density",
        "word_count",
        "faithfulness_score",
    ]

    summary_fields = [
        "agent",
        "total_dataset_cases",
        "successful_cases",
        "failed_cases",
        "skipped_cases",
        "successful_case_coverage",
        "average_flesch_kincaid",
        "flesch_kincaid_n",
        "average_smog",
        "smog_n",
        "average_jargon_density",
        "jargon_density_n",
        "average_word_count",
        "word_count_n",
        "average_faithfulness_score",
        "faithfulness_score_n",
        "refusal_rate",
        "refusal_n",
        "hallucination_rate",
        "hallucination_n",
    ]

    summary_rows = []

    all_results = list(results_by_key.values())

    for pipeline in AGENTS:

        rows = [
            row for row in all_results
            if row["pipeline"] == pipeline
            and row["model"] == MODEL
        ]

        successful_rows = [
            row for row in rows
            if row.get("status") == "success"
        ]

        failed_cases = sum(
            row.get("status") == "failed"
            for row in rows
        )

        skipped_cases = sum(
            row.get("status") == "skipped"
            for row in rows
        )

        summary = {
            "agent": pipeline,
            "total_dataset_cases": total_cases,
            "successful_cases": len(successful_rows),
            "failed_cases": failed_cases,
            "skipped_cases": skipped_cases,
            "successful_case_coverage": round(
                len(successful_rows) / total_cases, 4
            ) if total_cases else 0,
        }

        print(f"\n========== {pipeline} AVERAGES ==========")

        print(
            f"Successful: {len(successful_rows)}/{total_cases}"
        )

        print(f"Failed: {failed_cases}")
        print(f"Skipped: {skipped_cases}")

        for metric in metric_columns:

            average, count = calculate_average(
                successful_rows,
                metric,
            )

            summary[f"average_{metric}"] = (
                round(average, 4)
                if average is not None else None
            )

            summary[f"{metric}_n"] = count

            print(
                f"Average {metric}: "
                f"{round(average, 4) if average is not None else 'N/A'} "
                f"(n={count})"
            )

        refusal_rate, refusal_n = calculate_rate(
            successful_rows,
            "refusal",
        )

        hallucination_rate, hallucination_n = calculate_rate(
            successful_rows,
            "hallucination",
        )

        summary["refusal_rate"] = (
            round(refusal_rate, 4)
            if refusal_rate is not None else None
        )

        summary["refusal_n"] = refusal_n

        summary["hallucination_rate"] = (
            round(hallucination_rate, 4)
            if hallucination_rate is not None else None
        )

        summary["hallucination_n"] = hallucination_n

        print(
            f"Refusal rate: {summary['refusal_rate']} "
            f"(n={refusal_n})"
        )

        print(
            f"Hallucination rate: {summary['hallucination_rate']} "
            f"(n={hallucination_n})"
        )

        summary_rows.append(summary)

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n======================================")
    print("INDIVIDUAL AGENT AVERAGES SAVED")
    print("======================================")
    print(f"Summary: {SUMMARY_FILE}")


# --------------------------------------------------
# DATASET EVALUATION
# --------------------------------------------------

async def main():

    cases = load_test_cases()[:LIMIT]

    print("\n======================================")
    print("FAITHFULMED INDIVIDUAL AGENT EVALUATION")
    print("======================================")

    print(f"Loaded cases: {len(cases)}")
    print(f"Model: {MODEL}")
    print(f"Raw results: {OUTPUT_FILE}")
    print(f"Average results: {SUMMARY_FILE}")

    # Load previous results instead of clearing the CSV.
    load_existing_results()

    for index, case in enumerate(cases, start=1):

        example_id = str(case["case_id"])

        print(
            f"\n========== PROCESSING "
            f"{index}/{len(cases)} "
            f"(CASE {example_id}) =========="
        )

        await evaluate_case(case)

        print(
            f"\n[PROGRESS] Completed case "
            f"{index}/{len(cases)}"
        )

        await asyncio.sleep(CASE_DELAY_SECONDS)

    # Generate summary from all saved results.
    save_summary(
        total_cases=len(cases),
    )

    print("\n======================================")
    print("EVALUATION COMPLETE")
    print("======================================")

    print(f"Dataset cases: {len(cases)}")
    print(f"Model: {MODEL}")
    print(f"Individual results: {OUTPUT_FILE}")
    print(f"Agent averages: {SUMMARY_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
