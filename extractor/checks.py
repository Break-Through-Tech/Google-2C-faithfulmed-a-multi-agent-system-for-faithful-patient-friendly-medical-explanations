"""Deterministic post-processing of LLM output. No API calls."""
from __future__ import annotations

import re

from extractor.preprocess import normalize_for_match
from extractor.schema import Atom, AtomDraft


def assign_ids(drafts: list[AtomDraft]) -> list[Atom]:
    """A1..An in output order."""
    return [Atom(id=f"A{i}", **d.model_dump()) for i, d in enumerate(drafts, start=1)]


def check_evidence(atoms: list[Atom], source: str) -> list[str]:
    """Ids of atoms whose evidence is empty or not a substring of the source.

    A failure means the quote was paraphrased or invented. It does not by itself prove the
    claim is unfaithful, but it is a cheap, LLM-free signal worth reporting.
    """
    src = normalize_for_match(source)
    return [atom.id for atom in atoms if not _quote_in_source(atom.evidence, src)]


# "Mutations can cause... extra teeth": an elided quote. Accepted only if every fragment is
# verbatim and they appear in source order, so nothing can be invented across the gap.
# Added for gemini-3.5-flash-lite on dev q117 (before the test-set lock).
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|\u2026)\s*")


def _quote_in_source(evidence: str, normalized_source: str) -> bool:
    fragments = [f.strip().rstrip(".") for f in _ELLIPSIS.split(normalize_for_match(evidence))]
    fragments = [f for f in fragments if f]
    if not fragments:
        return False
    pos = 0
    for frag in fragments:
        found = normalized_source.find(frag, pos)
        if found < 0:
            return False
        pos = found + len(frag)
    return True


# Hedge words that, if present in an atom's evidence, contradict hedge="asserted".
_HEDGE_WORDS = re.compile(
    r"\b(may|might|can|could|should|often|sometimes|rarely|rare|usually|commonly|"
    r"frequently|possibly|probably|likely|unlikely|mostly|generally|typically|"
    r"occasionally|seldom|suggests?|not|never)\b",
    re.IGNORECASE,
)


def check_hedges(atoms: list[Atom]) -> list[str]:
    """Ids of atoms labelled 'asserted' whose evidence contains a hedge or negation word.

    Catches the "may" -> definite upgrade without an LLM. It is a flag, not a verdict:
    when the evidence spans a list, the hedge word may belong to a neighbouring clause.
    """
    return [a.id for a in atoms if a.hedge == "asserted" and _HEDGE_WORDS.search(a.evidence)]
