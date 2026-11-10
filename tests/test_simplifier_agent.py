"""
Tests for notebooks/simplifier_agent.py.

The offline tests use a fake Gemini client and a fake retriever, so they
spend no tokens. Each test prints what it checks and, for every check,
the expected vs actual value:

    python tests/test_simplifier_agent.py

The live test makes exactly 2 small Gemini calls and is skipped unless
RUN_LIVE_GEMINI=1. It uses the real MedlinePlus index when QDRANT_URL is
set, and canned definitions otherwise:

    RUN_LIVE_GEMINI=1 python tests/test_simplifier_agent.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"

if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))

import simplifier_agent as simplifier  # noqa: E402
from retrieval.retriever import RetrievalResult  # noqa: E402


RUN_LIVE = os.getenv("RUN_LIVE_GEMINI") == "1"

ATOMS = [
    {"text": "Bilateral lower-extremity edema", "type": "diagnosis"},
    {"text": "Follow up in 2 weeks", "type": "instruction"},
    {"text": "Furosemide 20 mg daily", "type": "medication"},
]


# Keep the report in order with unittest's summary when output is piped.
sys.stdout.reconfigure(line_buffering=True)


def definition_order_suite(module_name: str) -> unittest.TestSuite:
    """Run a module's tests in the order they're written, not alphabetically."""

    module = sys.modules[module_name]
    suite = unittest.TestSuite()

    for cls in vars(module).values():
        if isinstance(cls, type) and issubclass(cls, unittest.TestCase) and cls.__module__ == module_name:
            suite.addTests(cls(name) for name in vars(cls) if name.startswith("test"))

    return suite


def _short(value, limit: int = 100) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class ReadableTestCase(unittest.TestCase):
    """
    Prints each test's purpose and a ✓/✗ line per check with expected vs
    actual values, so the run reads as a report.
    """

    def setUp(self) -> None:
        name = self.id().split(".")[-1].removeprefix("test_").replace("_", " ")
        print(f"\n▶ {type(self).__name__}: {name}")

        if self._testMethodDoc:
            print("  What:", " ".join(self._testMethodDoc.split()))

    def _report(self, label: str, expected: str, actual: str, ok: bool) -> None:
        print(f"  {'✓' if ok else '✗'} {label}")
        print(f"      expected: {expected}")
        print(f"      actual:   {actual}")

        if not ok:
            self.fail(f"{label}: expected {expected}, got {actual}")

    def check(self, label: str, expected, actual) -> None:
        self._report(label, _short(expected), _short(actual), expected == actual)

    def check_true(self, label: str, actual) -> None:
        self.check(label, True, bool(actual))

    def check_in(self, label: str, needle: str, haystack: str, present: bool = True) -> None:
        found = needle in haystack
        self._report(
            label,
            f"{'contains' if present else 'does not contain'} {_short(needle)}",
            "found" if found else "not found",
            found == present,
        )

    def check_raises(self, label: str, error: type[Exception], pattern: str, fn) -> None:
        try:
            fn()
        except error as caught:
            actual, ok = f"{error.__name__}: {caught}", bool(re.search(pattern, str(caught)))
        else:
            actual, ok = "no error raised", False

        self._report(label, f"{error.__name__} matching {pattern!r}", _short(actual), ok)


def make_result(title: str, similarity: float, text: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        id=f"medlineplus:{title}",
        parent_id=f"medlineplus:{title}",
        title=title,
        text=text or f"{title} explained in plain words.",
        source="MedlinePlus",
        source_type="health_topic",
        url=f"https://medlineplus.gov/{title.lower().replace(' ', '')}.html",
        section="full_summary",
        chunk_index=0,
        distance=1.0 - similarity,
        similarity=similarity,
    )


class FakeRetriever:
    """Returns one strong and one weak match per term and records queries."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def __call__(self, term: str, k: int) -> list[RetrievalResult]:
        self.queries.append(term)
        return [make_result(term.title(), 0.8), make_result("Unrelated", 0.3)][:k]


def canned_retriever(term: str, k: int) -> list[RetrievalResult]:
    """Fixed lay definitions for the live test when Qdrant isn't configured."""

    definitions = {
        "edema": "Edema is swelling caused by fluid trapped in your body's tissues.",
        "furosemide": "Furosemide is a water pill. It helps your body get rid of extra fluid.",
    }

    for key, text in definitions.items():
        if key in term.lower():
            return [make_result(key.title(), 0.9, text)]

    return []


class FakeGeminiClient:
    """
    Stands in for genai.Client.

    JSON-mode calls (the Simplifier) get queued responses. Plain calls (the
    Readability Agent) get `plain_reply`.
    """

    def __init__(self, json_replies: list | None = None, plain_reply: str = "") -> None:
        self.json_replies = list(json_replies or [])
        self.plain_reply = plain_reply
        self.calls: list[dict] = []
        self.models = self

    def generate_content(self, model, contents, config=None):
        json_mode = config is not None and config.response_mime_type == "application/json"
        self.calls.append({"model": model, "contents": contents, "json_mode": json_mode})

        if json_mode:
            return SimpleNamespace(text=json.dumps(self.json_replies.pop(0)))

        return SimpleNamespace(text=self.plain_reply)

    def close(self) -> None:
        pass


def simplifier_replies(atoms: list[dict], terms: list[list[str]], plain: list[str]) -> list:
    """The two JSON replies the Simplifier expects: terms, then rewrites."""

    return [
        [{"index": i, "terms": t} for i, t in enumerate(terms)],
        [{"index": i, "plain_text": p} for i, p in enumerate(plain)],
    ]


class ParseAtomsTest(ReadableTestCase):
    def test_accepts_fenced_json_string(self) -> None:
        """The Extractor's output arrives as a ```json-fenced string; it should parse to the atom list."""

        raw = "```json\n" + json.dumps(ATOMS) + "\n```"
        self.check("parsed atoms", ATOMS, simplifier.parse_atoms(raw))

    def test_accepts_plain_strings_and_drops_empty_atoms(self) -> None:
        """Bare strings become {text, type: ""} atoms and blank atoms are dropped."""

        parsed = simplifier.parse_atoms(["Edema", {"text": "  "}, {"text": "Gout", "type": "diagnosis"}])
        self.check(
            "parsed atoms",
            [{"text": "Edema", "type": ""}, {"text": "Gout", "type": "diagnosis"}],
            parsed,
        )

    def test_rejects_empty_input(self) -> None:
        """An empty atom list (in any form) is an error, not an empty draft."""

        for bad in ([], "[]", [{"text": ""}]):
            self.check_raises(f"input {bad!r}", ValueError, ".", lambda: simplifier.parse_atoms(bad))


class ExtractTermsTest(ReadableTestCase):
    def test_ignores_out_of_range_indices_and_blank_terms(self) -> None:
        """Gemini's term list is cleaned: blank terms and indices for atoms that don't exist are ignored."""

        client = FakeGeminiClient([[
            {"index": 0, "terms": ["edema", " "]},
            {"index": 7, "terms": ["ghost"]},
        ]])

        terms = simplifier.extract_terms(ATOMS, client)

        self.check("terms per atom", [["edema"], [], []], terms)
        self.check("Gemini called in JSON mode", True, client.calls[0]["json_mode"])


class LookupDefinitionsTest(ReadableTestCase):
    def test_deduplicates_terms_case_insensitively(self) -> None:
        """A term that appears twice (any casing) is looked up in MedlinePlus only once."""

        retriever = FakeRetriever()
        simplifier.lookup_definitions(["Edema", "edema", "furosemide"], retriever)

        self.check("retrieval queries", ["Edema", "furosemide"], retriever.queries)

    def test_drops_matches_below_min_similarity(self) -> None:
        """Retrieved chunks below MIN_SIMILARITY are discarded so weak matches can't mislead Gemini."""

        definitions = simplifier.lookup_definitions(["edema"], FakeRetriever())

        print(f"  (MIN_SIMILARITY = {simplifier.MIN_SIMILARITY}; fake hits: Edema 0.8, Unrelated 0.3)")
        self.check("definitions kept", ["Edema"], [hit.title for hit in definitions["edema"]])

    def test_uses_module_retrieve_by_default(self) -> None:
        """With no retriever passed, the module's real `retrieve` is used (here patched with a fake)."""

        retriever = FakeRetriever()

        with patch.object(simplifier, "retrieve", retriever):
            simplifier.lookup_definitions(["edema"])

        self.check("retrieval queries", ["edema"], retriever.queries)


class RewriteAtomsTest(ReadableTestCase):
    def test_raises_when_an_atom_is_not_rewritten(self) -> None:
        """If Gemini skips any atom, the Simplifier errors instead of silently omitting a clinical fact."""

        client = FakeGeminiClient([[{"index": 0, "plain_text": "Swollen legs."}]])

        self.check_raises(
            "Gemini rewrote atom 0 only",
            ValueError,
            r"\[1, 2\]",
            lambda: simplifier.rewrite_atoms(ATOMS, [[], [], []], {}, client),
        )

    def test_prompt_marks_terms_without_definitions(self) -> None:
        """The rewrite prompt tells Gemini when a term has no definition and when an atom has no hard terms."""

        client = FakeGeminiClient([[{"index": i, "plain_text": "x"} for i in range(3)]])
        simplifier.rewrite_atoms(ATOMS, [["edema"], [], []], {}, client)

        prompt = client.calls[0]["contents"]
        self.check_in("prompt flags missing definition", '"edema": no definition found', prompt)
        self.check_in("prompt flags atoms with no hard terms", "(no difficult terms)", prompt)


class ProcessSimplificationTest(ReadableTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.plain = ["Both legs are swollen.", "Come back in 2 weeks.", "Take furosemide 20 mg every day."]
        self.client = FakeGeminiClient(simplifier_replies(
            ATOMS, [["lower-extremity edema"], [], ["Furosemide"]], self.plain,
        ))
        self.retriever = FakeRetriever()
        self.result = simplifier.process_simplification(ATOMS, self.client, retriever=self.retriever)

    def test_uses_exactly_two_gemini_calls(self) -> None:
        """A full run costs 2 Gemini requests (terms + rewrite), no matter how many atoms there are."""

        self.check("Gemini calls", 2, len(self.client.calls))
        self.check("model used", {simplifier.MODEL}, {call["model"] for call in self.client.calls})

    def test_draft_joins_atoms_in_source_order(self) -> None:
        """The final draft is the plain-language atoms joined in their original order."""

        self.check("draft", " ".join(self.plain), self.result["draft"])

    def test_rewrite_prompt_contains_retrieved_definitions(self) -> None:
        """Strong MedlinePlus matches reach the rewrite prompt; weak ones don't."""

        prompt = self.client.calls[1]["contents"]
        self.check_in(
            "strong match in prompt",
            '"lower-extremity edema" -> MedlinePlus "Lower-Extremity Edema"',
            prompt,
        )
        self.check_in("weak match in prompt", "Unrelated", prompt, present=False)

    def test_records_provenance_per_atom(self) -> None:
        """Each simplified atom keeps its source text and the MedlinePlus URL that grounded each term."""

        atom = self.result["simplified_atoms"][2]

        self.check("source text", "Furosemide 20 mg daily", atom["source_text"])
        self.check("plain text", self.plain[2], atom["plain_text"])
        self.check(
            "definition URL",
            "https://medlineplus.gov/furosemide.html",
            atom["definitions"]["Furosemide"][0]["url"],
        )


class SimplifierAdkCallbackTest(ReadableTestCase):
    def test_callback_returns_draft_and_saves_provenance(self) -> None:
        """Inside ADK, the callback reads state["atoms"], returns the draft, and saves provenance to state."""

        client = FakeGeminiClient(simplifier_replies(ATOMS, [[], [], []], ["a.", "b.", "c."]))
        context = SimpleNamespace(state={"atoms": json.dumps(ATOMS)})

        with patch("google.genai.Client", return_value=client), \
                patch.object(simplifier, "retrieve", FakeRetriever()):
            response = asyncio.run(simplifier.run_simplifier_adk(context, None))

        self.check("returned draft", "a. b. c.", response.content.parts[0].text)
        self.check("simplified_atoms saved", 3, len(context.state["simplified_atoms"]))


class TokenCountingClient:
    """Wraps a real genai.Client and totals the tokens each call used."""

    def __init__(self, client) -> None:
        self._client = client
        self.calls = 0
        self.total_tokens = 0
        self.models = self

    def generate_content(self, **kwargs):
        response = self._client.models.generate_content(**kwargs)
        self.calls += 1
        self.total_tokens += getattr(response.usage_metadata, "total_token_count", 0) or 0
        return response

    def close(self) -> None:
        self._client.close()


def live_retriever():
    """Real MedlinePlus index if configured; canned definitions otherwise."""

    return None if os.getenv("QDRANT_URL") else canned_retriever


@unittest.skipUnless(RUN_LIVE, "set RUN_LIVE_GEMINI=1 to call Gemini")
class LiveSimplifierTest(ReadableTestCase):
    def test_two_atoms_end_to_end(self) -> None:
        """Real Gemini simplifies 2 short atoms in 2 calls and keeps the dose unchanged."""

        if not os.getenv("GEMINI_API_KEY"):
            self.skipTest("GEMINI_API_KEY is not set (put .env in the repo root)")

        from google import genai

        # Two short atoms keep both prompts small.
        atoms = ATOMS[:1] + ATOMS[2:]
        client = TokenCountingClient(genai.Client())

        try:
            result = simplifier.process_simplification(atoms, client, retriever=live_retriever())
        finally:
            client.close()

        print(f"  Retrieval: {'MedlinePlus Qdrant index' if os.getenv('QDRANT_URL') else 'canned definitions'}")
        print(f"  Cost: {client.calls} Gemini calls, {client.total_tokens} tokens")
        print(f"  Draft: {result['draft']}")

        self.check("Gemini calls", 2, client.calls)
        self.check("simplified atoms", 2, len(result["simplified_atoms"]))
        self.check_in("dose kept verbatim", "20 mg", result["draft"])


def load_tests(loader, tests, pattern):
    return definition_order_suite(__name__)


if __name__ == "__main__":
    unittest.main(verbosity=0)
