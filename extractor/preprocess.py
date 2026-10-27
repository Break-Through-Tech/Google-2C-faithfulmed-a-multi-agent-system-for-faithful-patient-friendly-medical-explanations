"""Deterministic text cleanup applied before extraction and before evidence checks."""
from __future__ import annotations

import re

# MedAESQA citations: "[28646811]" or "[33659106, 12693795, 30204000]"
_PMID_BRACKETS = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")
_WHITESPACE = re.compile(r"\s+")
_CHAR_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                           "–": "-", "—": "-", " ": " "})


def preprocess(text: str) -> str:
    """Strip PMID citation brackets, normalize quotes/dashes, collapse whitespace."""
    text = text.translate(_CHAR_MAP)
    text = _PMID_BRACKETS.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_for_match(text: str) -> str:
    """Looser form used only for evidence substring matching."""
    return preprocess(text).casefold()
