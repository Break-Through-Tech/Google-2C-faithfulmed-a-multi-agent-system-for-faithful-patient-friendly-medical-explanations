

import asyncio
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------
# PROJECT SETUP
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

load_dotenv(ROOT / ".env")

from verifier import run_medaesqa_verifier


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = ROOT / "data" / "medaesqa_v1.json"
RESULTS_DIR = ROOT / "evaluation" / "results"

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Initial balanced test: five examples from each class.
SAMPLES_PER_CLASS = 20

# Makes the sample reproducible.
RANDOM_SEED = 42

SAFE_MODEL = re.sub(r"[^a-zA-Z0-9_-]", "_", MODEL)

# Separate output files preserve previous results.
OUTPUT_CSV = (
    RESULTS_DIR
    / f"verifier_sentence_predictions_balanced_{SAFE_MODEL}.csv"
)

OUTPUT_JSON = (
    RESULTS_DIR
    / f"verifier_sentence_calibration_metrics_balanced_{SAFE_MODEL}.json"
)

BINARY_LABELS = {"supporting", "contradicting"}


# --------------------------------------------------
# CSV FIELDS
# --------------------------------------------------

FIELDS = [
    "sentence_key",
    "input_fingerprint",
    "verifier_model",
    "question_id",
    "model_id",
    "sentence_id",
    "sentence_index",
    "sentence_text",
    "evidence_text",
    "evidence_relations",
    "evidence_excerpt_count",
    "missing_evidence_excerpt_count",
    "human_label",
    "verifier_faithful",
    "unsupported_claims",
    "omissions",
    "status",
    "error",
]


# --------------------------------------------------
# INPUT FINGERPRINT
# --------------------------------------------------

def create_fingerprint(
    sentence_text,
    evidence_text,
    relations,
    human_label,
    model,
):
    """
    Identify the exact inputs used for an API evaluation.

    A previous prediction is reusable only if these inputs
    still match.
    """

    payload = {
        "sentence_text": sentence_text,
        "evidence_text": evidence_text,
        "relations": relations,
        "human_label": human_label,
        "model": model,
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# SAVE AND LOAD RESULTS
# --------------------------------------------------

def save_predictions(records):
    """Save results incrementally."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        writer.writeheader()
        writer.writerows(records.values())


def load_predictions():
    """Load previous results for the current model."""

    records = {}

    if not OUTPUT_CSV.exists():
        return records

    with open(
        OUTPUT_CSV,
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != FIELDS:
            raise ValueError(
                "Existing balanced CSV has an incompatible format. "
                "Preserve it and use a new output filename."
            )

        for row in reader:

            if row.get("verifier_model") == MODEL:
                records[row["sentence_key"]] = row

    return records


# --------------------------------------------------
# DATASET PREPARATION
# --------------------------------------------------

def prepare_samples(dataset):

    samples = []
    exclusions = Counter()
    label_counts = Counter()

    for question_index, question in enumerate(dataset):

        question_id = question.get(
            "question_id",
            question.get("id", f"index_{question_index}")
        )

        answers = question.get("machine_generated_answers") or {}

        for model_id, answer in answers.items():

            sentences = answer.get("answer_sentences") or []

            for sentence_index, sentence in enumerate(sentences):

                sentence_text = (
                    sentence.get("answer_sentence") or ""
                ).strip()

                if not sentence_text:
                    exclusions["empty_sentence"] += 1
                    continue

                assessments = (
                    sentence.get("citation_assessment") or []
                )

                if not assessments:
                    exclusions["no_citation_assessments"] += 1
                    continue

                relations = [
                    (citation.get("evidence_relation") or "")
                    .strip()
                    .lower()
                    for citation in assessments
                ]

                label_counts.update(relations)

                # Every citation must have a binary label.
                if any(
                    relation not in BINARY_LABELS
                    for relation in relations
                ):
                    exclusions["non_binary_citation_label"] += 1
                    continue

                # All citations must agree on the label.
                unique_relations = set(relations)

                if len(unique_relations) != 1:
                    exclusions["mixed_citation_labels"] += 1
                    continue

                human_relation = relations[0]

                # Collect all available evidence excerpts.
                evidence_excerpts = []
                missing_excerpt_count = 0

                for citation in assessments:

                    excerpt = (
                        citation.get("evidence_support") or ""
                    ).strip()

                    if excerpt:
                        evidence_excerpts.append(excerpt)
                    else:
                        missing_excerpt_count += 1

                if not evidence_excerpts:
                    exclusions["no_available_evidence"] += 1
                    continue

                if missing_excerpt_count:
                    exclusions["incomplete_evidence"] += 1

                evidence_text = "\n\n".join(
                    f"Evidence {i}: {excerpt}"
                    for i, excerpt in enumerate(
                        evidence_excerpts,
                        start=1,
                    )
                )

                sentence_id = sentence.get(
                    "answer_sentence_id",
                    sentence_index + 1,
                )

                human_label = human_relation == "supporting"

                # Position prevents duplicate sentence IDs
                # from overwriting one another.
                sentence_key = (
                    f"{question_id}|{model_id}|"
                    f"{sentence_index}|{sentence_id}"
                )

                fingerprint = create_fingerprint(
                    sentence_text=sentence_text,
                    evidence_text=evidence_text,
                    relations=relations,
                    human_label=human_label,
                    model=MODEL,
                )

                samples.append({
                    "sentence_key": sentence_key,
                    "input_fingerprint": fingerprint,
                    "verifier_model": MODEL,
                    "question_id": question_id,
                    "model_id": model_id,
                    "sentence_id": sentence_id,
                    "sentence_index": sentence_index,
                    "sentence_text": sentence_text,
                    "evidence_text": evidence_text,
                    "evidence_relations": json.dumps(relations),
                    "evidence_excerpt_count": len(evidence_excerpts),
                    "missing_evidence_excerpt_count": missing_excerpt_count,
                    "human_label": human_label,
                })

    return samples, exclusions, label_counts


# --------------------------------------------------
# BALANCED SAMPLING
# --------------------------------------------------

def select_balanced_samples(samples):

    supporting_samples = [
        sample for sample in samples
        if sample["human_label"] is True
    ]

    contradicting_samples = [
        sample for sample in samples
        if sample["human_label"] is False
    ]

    rng = random.Random(RANDOM_SEED)

    selected_supporting = rng.sample(
        supporting_samples,
        min(SAMPLES_PER_CLASS, len(supporting_samples)),
    )

    selected_contradicting = rng.sample(
        contradicting_samples,
        min(SAMPLES_PER_CLASS, len(contradicting_samples)),
    )

    selected = selected_supporting + selected_contradicting

    rng.shuffle(selected)

    return (
        selected,
        len(supporting_samples),
        len(contradicting_samples),
        len(selected_supporting),
        len(selected_contradicting),
    )


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(rows):

    successful = [
        row for row in rows
        if row.get("status") == "success"
    ]

    if not successful:
        return {
            "sample_size": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "specificity": None,
            "balanced_accuracy": None,
            "f1": None,
            "cohen_kappa": None,
        }

    tp = tn = fp = fn = 0

    for row in successful:

        human = (
            row["human_label"].strip().lower() == "true"
        )

        predicted = (
            row["verifier_faithful"].strip().lower() == "true"
        )

        if human and predicted:
            tp += 1
        elif not human and not predicted:
            tn += 1
        elif not human and predicted:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else None
    )

    balanced_accuracy = (
        (recall + specificity) / 2
        if specificity is not None
        else None
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    # Cohen's kappa
    actual_positive = tp + fn
    actual_negative = tn + fp

    predicted_positive = tp + fp
    predicted_negative = tn + fn

    expected_agreement = (
        actual_positive * predicted_positive
        + actual_negative * predicted_negative
    ) / (total * total)

    if expected_agreement == 1:
        kappa = None
    else:
        kappa = (
            (accuracy - expected_agreement)
            / (1 - expected_agreement)
        )

    return {
        "sample_size": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "cohen_kappa": kappa,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


# --------------------------------------------------
# VERIFIER EVALUATION
# --------------------------------------------------

async def evaluate_samples(
    samples,
    exclusions,
    label_counts,
    selected_samples,
    population_counts,
    selected_class_counts,
):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records = load_predictions()

    # Reuse a previous result only if:
    # 1. The sentence key matches.
    # 2. The model matches.
    # 3. The input fingerprint matches.
    # 4. The previous evaluation succeeded.

    remaining = []

    for sample in selected_samples:

        previous = records.get(sample["sentence_key"])

        reusable = (
            previous is not None
            and previous.get("status") == "success"
            and previous.get("verifier_model") == MODEL
            and previous.get("input_fingerprint")
                == sample["input_fingerprint"]
        )

        if not reusable:
            remaining.append(sample)

    print("\n========== EVALUATION SETUP ==========")
    print("Eligible sentence samples:", len(samples))
    print("Available supporting:", population_counts[0])
    print("Available contradicting:", population_counts[1])
    print("Selected supporting:", selected_class_counts[0])
    print("Selected contradicting:", selected_class_counts[1])
    print("Selected total:", len(selected_samples))
    print("Previously completed and reusable:",
          len(selected_samples) - len(remaining))
    print("Remaining:", len(remaining))
    print("Verifier model:", MODEL)

    successful_this_run = 0
    failed_this_run = 0

    for index, sample in enumerate(remaining, start=1):

        print(
            f"\n[{index}/{len(remaining)}] "
            f"Question {sample['question_id']} "
            f"Model {sample['model_id']} "
            f"Sentence {sample['sentence_id']}"
        )

        try:

            verdict = await run_medaesqa_verifier(
                evidence_text=sample["evidence_text"],
                answer_sentence=sample["sentence_text"],
                model=MODEL,
            )

            if not isinstance(verdict, dict):
                raise TypeError(
                    "Expected the Verifier to return a dictionary."
                )

            if "faithful" not in verdict:
                raise ValueError(
                    "Verifier response did not contain 'faithful'."
                )

            if not isinstance(verdict["faithful"], bool):
                raise TypeError(
                    "Verifier 'faithful' value must be a boolean."
                )

            unsupported_claims = (
                verdict.get("unsupported_claims") or []
            )

            omissions = verdict.get("omissions") or []

            row = {
                **sample,
                "human_label": str(sample["human_label"]),
                "verifier_faithful": str(verdict["faithful"]),
                "unsupported_claims": json.dumps(
                    unsupported_claims,
                    ensure_ascii=False,
                ),
                "omissions": json.dumps(
                    omissions,
                    ensure_ascii=False,
                ),
                "status": "success",
                "error": "",
            }

            records[sample["sentence_key"]] = row
            successful_this_run += 1

            print(
                "Human:",
                "supporting" if sample["human_label"] else "contradicting",
                "| Verifier:",
                verdict["faithful"],
                "| Unsupported claims:",
                len(unsupported_claims),
            )

        except Exception as error:

            row = {
                **sample,
                "human_label": str(sample["human_label"]),
                "verifier_faithful": "",
                "unsupported_claims": "[]",
                "omissions": "[]",
                "status": "failed",
                "error": str(error),
            }

            records[sample["sentence_key"]] = row
            failed_this_run += 1

            print("Evaluation failed:", error)

        # Save after every API evaluation.
        save_predictions(records)

    # --------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------

    selected_keys = {
        sample["sentence_key"]
        for sample in selected_samples
    }

    selected_rows = [
        records[key]
        for key in selected_keys
        if key in records
        and records[key].get("input_fingerprint")
            == next(
                sample["input_fingerprint"]
                for sample in selected_samples
                if sample["sentence_key"] == key
            )
    ]

    metrics = calculate_metrics(selected_rows)

    successful_rows = [
        row for row in selected_rows
        if row.get("status") == "success"
    ]

    final_results = {
        "verifier_model": MODEL,
        "random_seed": RANDOM_SEED,
        "samples_per_class_requested": SAMPLES_PER_CLASS,
        "evaluation_unit": (
            "One answer sentence evaluated against all available "
            "citation evidence for that sentence."
        ),
        "human_label_mapping": {
            "supporting": True,
            "contradicting": False,
        },
        "eligible_sentence_samples": len(samples),
        "available_supporting_sentences": population_counts[0],
        "available_contradicting_sentences": population_counts[1],
        "selected_supporting_sentences": selected_class_counts[0],
        "selected_contradicting_sentences": selected_class_counts[1],
        "selected_sentence_samples": len(selected_samples),
        "successful_evaluations": len(successful_rows),
        "failed_evaluations": sum(
            row.get("status") == "failed"
            for row in selected_rows
        ),
        "successful_evaluations_this_run": successful_this_run,
        "failed_evaluations_this_run": failed_this_run,
        "human_citation_label_counts": dict(label_counts),
        "excluded_sentence_counts": dict(exclusions),
        "sentence_level_metrics": metrics,
        "note": (
            "This is a reproducible balanced pilot with up to five "
            "sentences per class. Each sentence is evaluated once "
            "against its combined available citation evidence. "
            "Mixed and non-binary citation labels are excluded. "
            "The balanced sample does not represent the original "
            "class distribution. Results are sentence-level, not "
            "whole-answer accuracy measurements."
        ),
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n========== FINAL CALIBRATION RESULTS ==========")
    print(json.dumps(final_results, indent=4))

    print("\nPredictions:", OUTPUT_CSV)
    print("Metrics:", OUTPUT_JSON)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

async def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    with open(DATA_PATH, encoding="utf-8") as file:
        dataset = json.load(file)

    samples, exclusions, label_counts = prepare_samples(dataset)

    (
        selected_samples,
        support_count,
        contradict_count,
        selected_support,
        selected_contradict,
    ) = select_balanced_samples(samples)

    print("\nHuman citation label counts:")
    print(json.dumps(dict(label_counts), indent=4))

    print("\nExcluded sentence counts:")
    print(json.dumps(dict(exclusions), indent=4))

    print("\nSentence-level population:")
    print("Supporting:", support_count)
    print("Contradicting:", contradict_count)

    await evaluate_samples(
        samples=samples,
        exclusions=exclusions,
        label_counts=label_counts,
        selected_samples=selected_samples,
        population_counts=(
            support_count,
            contradict_count,
        ),
        selected_class_counts=(
            selected_support,
            selected_contradict,
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
