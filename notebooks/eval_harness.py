"""
FaithfulMed — starter evaluation harness.

Build this out in Week 1-2, BEFORE the agents. You can't improve what you can't measure.
Run: python notebooks/eval_harness.py  (expects data/medaesqa_v1.json — see data/README.md)

Provides:
  - readability_scores(text): Flesch-Kincaid grade, SMOG, jargon density, length
  - meets_readability_target(text): the <= 8th-grade success criterion
  - load_medaesqa(path): loads the gold eval set
  - verifier_agreement(human, verifier): accuracy + Cohen's kappa for Verifier calibration

This is model-agnostic — it scores text, whatever produced it. Extend, don't treat as final.
Each agent owner plugs their agent's outputs into this harness.
"""

from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path

import textstat
from sklearn.metrics import cohen_kappa_score, accuracy_score

import os
import re

import requests
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score, accuracy_score

load_dotenv()
UMLS_API_KEY = os.getenv("UMLS_API_KEY")

DATA = Path(__file__).resolve().parent.parent / "data" / "medaesqa_v1.json"

@lru_cache(maxsize=5000)
def umls_lookup(term: str) -> bool:
    """Return True if UMLS recognizes the term as a medical concept."""

    if not UMLS_API_KEY:
        raise RuntimeError("UMLS_API_KEY is not set.")

    url = "https://uts-ws.nlm.nih.gov/rest/search/current"

    params = {
        "string": term,
        "apiKey": UMLS_API_KEY,
        "returnIdType": "concept",
        "pageSize": 1,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        return len(data.get("result", {}).get("results", [])) > 0

    except requests.RequestException as e:
        print(f"UMLS lookup failed for '{term}': {e}")
        return False


def generate_phrases(words: list[str], max_length: int = 3) -> list[str]:
    """Generate 1-, 2-, and 3-word phrases."""
    
    phrases = []

    for length in range(1, max_length + 1):
        for i in range(len(words) - length + 1):
            phrases.append(" ".join(words[i:i + length]))

    return phrases


def calculate_umls_jargon_density(text: str) -> float:
    """Estimate medical-term density using UMLS."""

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    if not words:
        return 0.0

    medical_word_count = 0
    i = 0

    while i < len(words):
        matched = False

        # Try 3-word phrases first
        if i + 3 <= len(words):
            phrase = " ".join(words[i:i + 3])

            if umls_lookup(phrase):
                medical_word_count += 3
                i += 3
                matched = True

        # If no 3-word match, try 2-word phrase
        if not matched and i + 2 <= len(words):
            phrase = " ".join(words[i:i + 2])

            if umls_lookup(phrase):
                medical_word_count += 2
                i += 2
                matched = True

        # If no phrase match, try 1 word
        if not matched:
            if umls_lookup(words[i]):
                medical_word_count += 1

            i += 1

    return medical_word_count / len(words)

def readability_scores(text: str) -> dict:
    """Calculate readability and medical-term metrics."""

    words = max(textstat.lexicon_count(text, removepunct=True), 1)

    return {
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "smog_index": textstat.smog_index(text),
        "medical_term_density": calculate_umls_jargon_density(text),
        "word_count": words,
    }


def meets_readability_target(text: str, max_grade: float = 8.0) -> bool:
    """Success criterion: <= 8th-grade reading level (Flesch-Kincaid)."""
    return textstat.flesch_kincaid_grade(text) <= max_grade


def load_medaesqa(path: Path = DATA) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download medaesqa_v1.json from https://osf.io/ydbzq "
            f"into the data/ folder (see data/README.md)."
        )
    with open(path) as f:
        return json.load(f)


def verifier_agreement(human_labels: list[int], verifier_labels: list[int]) -> dict:
    """Calibrate the Verifier against human faithfulness labels.

    Targets from the success criteria: accuracy >= 0.80, Cohen's kappa >= 0.6.
    Pass 0/1 (or categorical) labels of equal length.
    """
    return {
        "accuracy": accuracy_score(human_labels, verifier_labels),
        "cohen_kappa": cohen_kappa_score(human_labels, verifier_labels),
        "n": len(human_labels),
    }


if __name__ == "__main__":
    demo = ("Your discharge summary says you were prescribed a beta-blocker to manage "
            "hypertension and should follow up with cardiology in two weeks.")
    print("Readability demo:", readability_scores(demo))
    print("Meets <=8th grade:", meets_readability_target(demo))

    # Tiny agreement demo (replace with real Verifier vs. human labels):
    print("Agreement demo:", verifier_agreement([1, 1, 0, 1, 0], [1, 0, 0, 1, 0]))

    try:
        data = load_medaesqa()
        print(f"Loaded MedAESQA: {len(data)} questions.")
    except FileNotFoundError as e:
        print(e)

