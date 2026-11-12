
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "medaesqa_v1.json"


def load_test_cases():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    cases = []

    for item in dataset:
        cases.append({
            "case_id": str(item["question_id"]),
            "question": item["question"],
            "question_frame": item.get("question_frame", {}),
            "source_text": item["expert_curated_answer"].strip(),
            "reference_answer": item["expert_curated_answer"].strip(),
            "machine_generated_answers": item.get(
                "machine_generated_answers", {}
            ),
            "nuggets": item.get("nuggets", [])
        })

    return cases


if __name__ == "__main__":
    cases = load_test_cases()

    print(f"Loaded {len(cases)} test cases.")

    for case in cases[:3]:
        print(
            f"\nCase {case['case_id']}: "
            f"{case['question']}"
        )
