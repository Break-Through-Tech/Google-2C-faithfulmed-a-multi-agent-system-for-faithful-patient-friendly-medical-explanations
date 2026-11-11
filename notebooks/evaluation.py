"""Reproducible scoring and reporting for Refiner fixture runs."""

from __future__ import annotations

import json
import math
import platform
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import textstat
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .agent import RefinerConfig
from .prompts import PROMPT_VERSION
from .schemas import EvaluationFixture, PredictionRecord, RefinerOutput

try:
    import nltk

    nltk.data.find("corpora/cmudict")
    _TEXTSTAT_SYLLABLES_READY = True
except (ImportError, LookupError):
    _TEXTSTAT_SYLLABLES_READY = False


def load_fixtures(path: str | Path) -> list[EvaluationFixture]:
    records: list[EvaluationFixture] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(EvaluationFixture.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"invalid fixture at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"fixture file is empty: {path}")
    ids = [record.example_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate example_id in {path}")
    return records


def select_diverse_fixtures(
    fixtures: list[EvaluationFixture], limit: int | None
) -> list[EvaluationFixture]:
    """Select a deterministic, category-balanced subset and interleave its categories.

    Fixtures are stored in category blocks, so taking ``fixtures[:limit]`` can erase most of the
    dataset's diversity. Within each category, evenly spaced examples preserve early and late
    template variants. Interleaving prevents a partial live run from covering only one category.
    """

    if limit is None or limit >= len(fixtures):
        return list(fixtures)
    if limit < 1:
        raise ValueError("limit must be at least 1")

    by_category: dict[str, list[EvaluationFixture]] = {}
    for fixture in fixtures:
        by_category.setdefault(fixture.category, []).append(fixture)

    categories = list(by_category)
    allocations = {category: limit // len(categories) for category in categories}
    for category in categories[: limit % len(categories)]:
        allocations[category] += 1

    selected_by_category: dict[str, list[EvaluationFixture]] = {}
    for category in categories:
        rows = by_category[category]
        count = min(allocations[category], len(rows))
        if count == 0:
            selected_by_category[category] = []
        elif count == 1:
            selected_by_category[category] = [rows[0]]
        else:
            indexes = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
            selected_by_category[category] = [rows[index] for index in indexes]

    selected: list[EvaluationFixture] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for category in categories:
            rows = selected_by_category[category]
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
        if not added:
            break
        depth += 1
    return selected


def readability_scores(text: str) -> dict[str, float | int]:
    words_list = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    words = max(len(words_list), 1)
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    try:
        if not _TEXTSTAT_SYLLABLES_READY:
            raise LookupError("NLTK cmudict unavailable")
        fk_grade = float(textstat.flesch_kincaid_grade(text))
        smog = float(textstat.smog_index(text))
    except LookupError:
        # textstat 0.7.13 may require the separately downloaded NLTK cmudict. Keep offline
        # evaluation deterministic instead of initiating a network download during a test.
        syllable_counts = [_approximate_syllables(word) for word in words_list]
        syllables = sum(syllable_counts)
        polysyllables = sum(count >= 3 for count in syllable_counts)
        fk_grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
        smog = 1.043 * math.sqrt(polysyllables * (30 / sentences)) + 3.1291
    jargon = {
        "dysuria", "hematuria", "hydronephrosis", "nephrolithiasis", "ureteral",
        "stricture", "tamsulosin", "oxybutynin", "nitrofurantoin", "sildenafil",
        "finasteride", "urology",
    }
    return {
        "flesch_kincaid_grade": fk_grade,
        "smog_score": smog,
        "jargon_density": sum(word.lower() in jargon for word in words_list) / words,
        "word_count": int(words),
    }


def _approximate_syllables(word: str) -> int:
    word = word.lower().strip("e")
    groups = re.findall(r"[aeiouy]+", word)
    return max(1, len(groups))


def score_prediction(
    fixture: EvaluationFixture,
    output: RefinerOutput,
    latency_seconds: float,
    token_usage: dict[str, Any] | None = None,
) -> PredictionRecord:
    corrections_met = sum(item in output.refined_text for item in fixture.expected_corrections)
    preserved = sum(item in output.refined_text for item in fixture.preserve_facts)
    false_flags = [
        flag
        for flag in fixture.verifier_report.flagged_issues
        if flag not in fixture.gold_flagged_issues
    ]
    # Per-flag attribution is unavailable in the minimal public contract. False-only cases provide
    # an exact rejection signal; mixed cases are not credited automatically.
    false_rejected = (
        len(false_flags)
        if false_flags
        and not fixture.gold_flagged_issues
        and output.refined_text == fixture.draft
        and not output.needs_refinement
        else 0
    )
    correction_success = (
        output.needs_refinement == fixture.gold_needs_refinement
        and corrections_met == len(fixture.expected_corrections)
        and set(output.unresolved_issues) == set(fixture.expected_unresolved_issues)
        and preserved == len(fixture.preserve_facts)
        and output.refined_text == fixture.gold_refined_text
    )
    return PredictionRecord(
        example_id=fixture.example_id,
        category=fixture.category,
        source_text=fixture.source_text,
        draft=fixture.draft,
        verifier_report=fixture.verifier_report,
        gold_needs_refinement=fixture.gold_needs_refinement,
        gold_refined_text=fixture.gold_refined_text,
        prediction=output,
        latency_seconds=latency_seconds,
        token_usage=token_usage,
        correction_success=correction_success,
        false_flags_rejected=false_rejected,
        false_flags_total=len(false_flags),
        required_corrections_met=corrections_met,
        required_corrections_total=len(fixture.expected_corrections),
        preserve_facts_retained=preserved,
        preserve_facts_total=len(fixture.preserve_facts),
        readability_before=readability_scores(fixture.draft),
        readability_after=readability_scores(output.refined_text),
    )


def _classification(gold: list[bool], predicted: list[bool]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "precision": float(precision_score(gold, predicted, zero_division=0)),
        "recall": float(recall_score(gold, predicted, zero_division=0)),
        "f1": float(f1_score(gold, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(gold, predicted, labels=[False, True]).tolist(),
        "always_yes_baseline_accuracy": sum(gold) / len(gold),
        "n": len(gold),
    }


def aggregate_metrics(records: list[PredictionRecord]) -> dict[str, Any]:
    successful = [record for record in records if record.prediction is not None]
    if not successful:
        raise ValueError("no successful predictions to score")
    gold = [record.gold_needs_refinement for record in successful]
    predicted = [record.prediction.needs_refinement for record in successful if record.prediction]
    by_category: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in successful:
        by_category[record.category].append(record)
    correction_total = sum(record.required_corrections_total for record in successful)
    correction_met = sum(record.required_corrections_met for record in successful)
    false_total = sum(record.false_flags_total for record in successful)
    false_rejected = sum(record.false_flags_rejected for record in successful)
    positive_cases = [record for record in successful if record.gold_needs_refinement]
    latencies = sorted(record.latency_seconds for record in successful)
    fk_after = [float(record.readability_after["flesch_kincaid_grade"]) for record in successful]
    usage_rows = [record.token_usage for record in successful if record.token_usage]
    total_tokens = sum(
        int(row.get("total_token_count", 0))
        for row in usage_rows
        if isinstance(row.get("total_token_count", 0), (int, float))
    )
    return {
        "classification": _classification(gold, predicted),
        "per_category": {
            category: _classification(
                [record.gold_needs_refinement for record in rows],
                [record.prediction.needs_refinement for record in rows if record.prediction],
            )
            for category, rows in sorted(by_category.items())
        },
        "correction_quality": {
            "flagged_issue_correction_rate": correction_met / correction_total if correction_total else None,
            "case_level_correction_success_rate": (
                sum(bool(record.correction_success) for record in positive_cases) / len(positive_cases)
                if positive_cases
                else None
            ),
            "false_flag_rejection_rate": false_rejected / false_total if false_total else None,
            "unflagged_edit_rate": (
                sum(
                    record.prediction.refined_text != record.gold_refined_text
                    for record in successful
                )
                / len(successful)
            ),
            "unflagged_edit_definition": (
                "exact-gold-output mismatch proxy on synthetic fixtures; inspect predictions for "
                "clinical interpretation"
            ),
            "unresolved_issue_case_rate": (
                sum(bool(record.prediction.unresolved_issues) for record in successful) / len(successful)
            ),
            "new_clinically_meaningful_error_rate": None,
            "new_error_rate_status": "pending independent clinical review",
            "faithfulness_before_after": None,
            "faithfulness_status": "pending independent clinical claim annotation/review",
            "clinically_important_omission_rate": None,
            "omission_status": "pending independent clinical review",
        },
        "readability": {
            "mean_flesch_kincaid_after": statistics.fmean(fk_after),
            "eighth_grade_pass_rate_after": sum(value <= 8 for value in fk_after) / len(fk_after),
            "note": "Tracked only; the Refiner is not authorized to optimize unflagged readability.",
        },
        "efficiency": {
            "mean_latency_seconds": statistics.fmean(latencies),
            "median_latency_seconds": statistics.median(latencies),
            "p95_latency_seconds": latencies[max(0, int(0.95 * len(latencies)) - 1)],
            "token_usage": {"total_token_count": total_tokens} if usage_rows else None,
            "token_usage_status": (
                "reported by ADK"
                if usage_rows
                else "not available for fixture-mock runs; captured by live runner when emitted"
            ),
        },
        "errors": len(records) - len(successful),
    }


def dependency_versions() -> dict[str, str]:
    packages = ["google-adk", "google-genai", "pydantic", "textstat", "scikit-learn"]
    return {package: metadata.version(package) for package in packages}


def build_report_metadata(
    *, dataset: str, run_type: str, config: RefinerConfig, count: int
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "dataset_version": "urology-synthetic-v2.0.0",
        "run_type": run_type,
        "prompt_version": PROMPT_VERSION,
        "model": config.model if run_type == "live-gemini" else "fixture-mock (no model call)",
        "model_configuration": {
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
            "request_timeout_seconds": config.request_timeout_seconds,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "dependencies": dependency_versions(),
        "examples": count,
        "clinical_annotation_status": "synthetic, provisional, not independently clinically reviewed",
    }


def write_artifacts(
    output_dir: str | Path,
    metadata_record: dict[str, Any],
    metrics: dict[str, Any],
    records: list[PredictionRecord],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata_record, "metrics": metrics}
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    classification = metrics["classification"]
    quality = metrics["correction_quality"]
    matrix = classification["confusion_matrix"]

    def rate(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.3f} ({value:.1%})"

    report = f"""# FaithfulMed Refiner Evaluation Report

## Run identity

- Run type: `{metadata_record['run_type']}`
- Dataset: `{metadata_record['dataset']}` ({metadata_record['dataset_version']})
- Examples attempted: {metadata_record['examples']}
- Successfully scored: {classification['n']}
- Errors: {metrics['errors']}
- Model: `{metadata_record['model']}`
- Prompt: `{metadata_record['prompt_version']}`
- Clinical labels: {metadata_record['clinical_annotation_status']}

## Binary decision metrics

**Binary decision** means the Refiner's YES/NO answer in `needs_refinement`; it does not by itself
show whether the corrected wording was good. Scores below use only successfully scored cases.

| Metric | Small definition | Value |
|---|---|---:|
| Accuracy | Share of all YES/NO decisions that were correct | {rate(classification['accuracy'])} |
| Precision | When the Refiner said YES, how often YES was correct | {rate(classification['precision'])} |
| Recall | Of cases that truly needed refinement, how many the Refiner found | {rate(classification['recall'])} |
| F1 | One score balancing precision and recall | {rate(classification['f1'])} |
| Always-YES baseline accuracy | Score from saying YES to every case; a simple comparison point | {rate(classification['always_yes_baseline_accuracy'])} |

**Confusion matrix** means the count of the four possible YES/NO outcomes:

| Expected answer | Refiner said NO | Refiner said YES |
|---|---:|---:|
| Expected NO | {matrix[0][0]} | {matrix[0][1]} |
| Expected YES | {matrix[1][0]} | {matrix[1][1]} |

The raw matrix, using rows/columns `[NO, YES]`, is `{matrix}`.

## Correction metrics

- **Flagged-issue correction rate** — share of individual required fixes found in the output:
  {rate(quality['flagged_issue_correction_rate'])}
- **Case-level correction success** — share of positive cases that met every strict expected-output
  requirement: {rate(quality['case_level_correction_success_rate'])}
- **False-flag rejection rate** — share of incorrect Verifier flags that the Refiner correctly
  rejected without changing the draft: {rate(quality['false_flag_rejection_rate'])}
- **Exact-gold mismatch proxy** (stored as `unflagged_edit_rate`) — share whose wording did not exactly
  match the expected or **gold** answer: {rate(quality['unflagged_edit_rate'])}
- **Unresolved-issue case rate** — share where the Refiner said at least one supplied flag could not
  be safely resolved from the source: {rate(quality['unresolved_issue_case_rate'])}

A **flag** is a specific problem reported by the Verifier. A **gold answer** is the expected test
answer used for grading and is never sent to Gemini. **Synthetic** means the examples were created
for testing rather than taken from real patient records.

The exact-gold mismatch proxy is strict: different wording can be reasonable and still count as a
mismatch. Inspect `predictions.jsonl` before treating a mismatch as a clinical error.

Faithfulness, clinically important omission, and new-clinical-error rates are **pending independent
clinical review**. They are not fabricated from the Refiner's own judgments.

## Interpretation

This report is a `{metadata_record['run_type']}` result. Fixture-mock metrics validate the harness
and contracts; they are not evidence of Gemini model quality. Readability is tracked, not optimized,
because unflagged readability editing is outside the Refiner's scope.
"""
    (output_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
