"""
Tests for notebooks/medlineplus_abbreviations.py, the exact-match
abbreviation table the Simplifier checks before vector retrieval.

No network or Gemini calls:

    python tests/test_medlineplus_abbreviations.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Lets `python tests/test_medlineplus_abbreviations.py` find the tests package.
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_simplifier_agent import ReadableTestCase, definition_order_suite  # noqa: E402

import medlineplus_abbreviations as abbr  # noqa: E402  (path set by the import above)


def stands_for(term: str) -> str | None:
    entry = abbr.lookup_abbreviation(term)
    return entry.stands_for if entry else None


class LookupTest(ReadableTestCase):
    def test_dosing_codes_match_in_any_case_and_with_dots(self) -> None:
        """BID/TID/QID/PRN are often written lower case or dotted in notes; every form should match."""

        for term in ("BID", "bid", "b.i.d.", "t.i.d.", "QID", "prn", "p.r.n."):
            entry = abbr.lookup_abbreviation(term)
            self.check(f"{term!r} found", True, entry is not None)

        self.check("BID means", "Twice a day", stands_for("bid"))
        self.check("PRN means", "As needed", stands_for("p.r.n."))

    def test_other_abbreviations_must_be_upper_case(self) -> None:
        """Only upper case matches, so ordinary words (cat, pad, pet) are never treated as abbreviations."""

        self.check("'CAT'", "Computerized axial tomography", stands_for("CAT"))
        self.check("'HTN'", "Hypertension", stands_for("HTN"))

        for word in ("cat", "pad", "pet", "Pet", "htn"):
            self.check(f"{word!r}", None, stands_for(word))

    def test_handles_plurals_and_punctuation(self) -> None:
        """Terms as Gemini copies them ("PVCs", "ECG,") still match."""

        self.check("'PVCs'", "Premature ventricular contraction", stands_for("PVCs"))
        self.check("'ECG,'", "Electrocardiogram", stands_for("ECG,"))
        self.check("'D&C'", "Dilatation and curettage", stands_for("D&C"))
        self.check("'CA-125'", "Cancer antigen 125", stands_for("CA-125"))

    def test_unlisted_abbreviations_return_none(self) -> None:
        """Abbreviations the source doesn't list are not guessed at."""

        for term in ("PO", "QD", "NPO", "s/p"):
            self.check(f"{term!r}", None, abbr.lookup_abbreviation(term))


class SourceFixesTest(ReadableTestCase):
    def test_ambiguous_entries_have_no_meaning(self) -> None:
        """Abbreviations with several clinical meanings are flagged and carry no expansion to pick from."""

        for term in abbr.AMBIGUOUS:
            entry = abbr.lookup_abbreviation(term)
            self.check(f"{term} ambiguous, no expansion", (True, None), (entry.ambiguous, entry.stands_for))

    def test_inaccurate_source_notes_are_dropped(self) -> None:
        """Notes found wrong in the source (e.g. ALT 'test for ALP') are removed; the expansion stays."""

        for term in ("ALT", "AST", "PET"):
            entry = abbr.lookup_abbreviation(term)
            self.check(f"{term} note", None, entry.note)

        self.check("ALT still expands", "Alanine aminotransferase", stands_for("ALT"))

    def test_corrected_expansion(self) -> None:
        """The source's 'Post-traumatic stress syndrome' is corrected to 'disorder'."""

        self.check("PTSD", "Post-traumatic stress disorder", stands_for("PTSD"))

    def test_every_override_names_a_real_entry(self) -> None:
        """Override lists can't silently refer to abbreviations that aren't in the table."""

        overrides = {*abbr.AMBIGUOUS, *abbr.NOTE_DROPPED, *abbr.STANDS_FOR_FIXED, *abbr.ANY_CASE}
        self.check("overrides missing from table", [], sorted(overrides - set(abbr.ABBREVIATIONS)))


class FindAbbreviationsTest(ReadableTestCase):
    def test_finds_each_listed_abbreviation_once_in_order(self) -> None:
        """Scanning an atom returns listed abbreviations as written, in order, without repeats or ordinary words."""

        text = "Metformin 500 mg b.i.d. for DM; HTN, HTN. Has a pet cat. History of MS."
        self.check("found", ["b.i.d", "DM", "HTN", "MS"], abbr.find_abbreviations(text))


def load_tests(loader, tests, pattern):
    return definition_order_suite(__name__)


if __name__ == "__main__":
    unittest.main(verbosity=0)
