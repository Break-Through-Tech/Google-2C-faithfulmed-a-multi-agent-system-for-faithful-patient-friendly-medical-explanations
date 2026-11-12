import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from eval_harness import readability_scores, meets_readability_target
from simplifier_agent import process_simplification

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Fixed, synthetic test case. Do not change between evaluation runs.
TEST_ATOMS = [
    {"text": "The patient has hypertension.", "type": "condition"},
    {
        "text": "The patient was prescribed lisinopril 10 mg once daily.",
        "type": "medication",
    },
    {
        "text": "The patient should monitor blood pressure at home.",
        "type": "instruction",
    },
    {
        "text": "The patient should follow up with their physician in two weeks.",
        "type": "instruction",
    },
]

# Details that must remain unchanged in the medication rewrite.
PROTECTED_DETAILS = ["lisinopril", "10 mg", "once daily"]


def evaluate():
    client = genai.Client()

    try:
        result = process_simplification(TEST_ATOMS, client)
    finally:
        client.close()

    draft = result["draft"]
    simplified_atoms = result["simplified_atoms"]

    # Readability
    readability = readability_scores(draft)
    meets_grade_target = meets_readability_target(draft)

    # Fact completion
    expected_count = len(TEST_ATOMS)
    completed_count = len(simplified_atoms)
    completion_rate = completed_count / expected_count

    # Medication detail preservation
    medication_rewrite = next(
        (
            atom["plain_text"]
            for atom in simplified_atoms
            if atom["type"] == "medication"
        ),
        "",
    )

    preserved_details = [
        detail
        for detail in PROTECTED_DETAILS
        if detail.lower() in medication_rewrite.lower()
    ]

    medication_preservation = (
        len(preserved_details) / len(PROTECTED_DETAILS)
    )

    # Retrieval coverage: percentage of identified terms with definitions.
    all_terms = [
        definition
        for atom in simplified_atoms
        for definition in atom["definitions"].values()
    ]

    terms_with_definitions = sum(bool(hits) for hits in all_terms)
    retrieval_coverage = (
        terms_with_definitions / len(all_terms) if all_terms else 1.0
    )

    report = {
        "agent": "Simplifier",
        "readability": readability,
        "meets_grade_8_target": meets_grade_target,
        "fact_completion_rate": completion_rate,
        "completed_facts": completed_count,
        "expected_facts": expected_count,
        "medication_detail_preservation": medication_preservation,
        "preserved_medication_details": preserved_details,
        "expected_medication_details": PROTECTED_DETAILS,
        "retrieval_coverage": retrieval_coverage,
        "simplified_atoms": simplified_atoms,
        "final_draft": draft,
    }

    output_path = ROOT / "evaluation" / "simplifier_results.json"
    output_path.write_text(json.dumps(report, indent=2))

    print(json.dumps({
        key: value
        for key, value in report.items()
        if key not in ("simplified_atoms", "final_draft")
    }, indent=2))

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    evaluate()
