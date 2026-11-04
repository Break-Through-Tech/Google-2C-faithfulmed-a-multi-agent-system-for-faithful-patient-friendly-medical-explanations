"""
Tests that the Simplifier's output feeds the Readability Agent.

Offline tests use a fake Gemini client and retriever, so they spend no
tokens. Each test prints what it checks, the readability scores of the text
it hands over, and expected vs actual for every check. One test runs both
agents in a real ADK SequentialAgent to check that the Simplifier's `draft`
reaches the Readability Agent through session state:

    python tests/test_simplifier_to_readability.py

The live test is skipped unless RUN_LIVE_GEMINI=1. It makes 2 Simplifier
calls plus at most 2 Readability rewrites:

    RUN_LIVE_GEMINI=1 python tests/test_simplifier_to_readability.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

# Lets `python tests/test_simplifier_to_readability.py` find the tests package.
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from google.adk.agents import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from tests.test_simplifier_agent import (
    ATOMS,
    RUN_LIVE,
    FakeGeminiClient,
    FakeRetriever,
    ReadableTestCase,
    TokenCountingClient,
    definition_order_suite,
    live_retriever,
    simplifier,
    simplifier_replies,
)

import eval_harness  # noqa: E402  (path set by the import above)
import readability_agent as readability  # noqa: E402


EASY_PLAIN = ["Both legs are swollen.", "Come back in 2 weeks.", "Take furosemide 20 mg each day."]

# A plain-atom rewrite that is still too hard, so Readability must step in.
HARD_PLAIN = [
    "The patient exhibits bilateral lower-extremity edema, characterized by "
    "interstitial fluid accumulation attributable to compromised venous return.",
    "Subsequent outpatient reevaluation is recommended within approximately two weeks.",
    "Pharmacological management consists of furosemide administered at 20 mg daily.",
]

READABLE_REWRITE = (
    "The patient has swelling in both legs. "
    "The patient should come back in 2 weeks. "
    "The patient takes furosemide 20 mg each day."
)

NO_TERMS = [[], [], []]

JARGON_METHOD = (
    "UMLS concept lookup" if eval_harness.UMLS_API_KEY
    else "textstat difficult-words fallback (no UMLS_API_KEY)"
)


def print_scoring_rule() -> None:
    print(f"  Scoring: pass = Flesch-Kincaid grade <= {readability.TARGET_GRADE} "
          f"(eval_harness.meets_readability_target).")
    print(f"           SMOG, medical-term density [{JARGON_METHOD}] and word count are reported only.")
    print(f"           Readability rewrites at most {readability.MAX_REWRITES} times and keeps the lowest grade.")


def print_scores(label: str, text: str, scores: dict) -> None:
    print(f"  {label}: {text}")
    print(f"      Flesch-Kincaid grade  {scores['flesch_kincaid_grade']:6.2f}   "
          f"{'PASS' if scores['meets_target'] else 'FAIL'} (target <= {readability.TARGET_GRADE})")
    print(f"      SMOG index            {scores['smog_index']:6.2f}")
    print(f"      Medical-term density  {scores['medical_term_density']:6.2f}")
    print(f"      Word count            {scores['word_count']:6d}")


class SimplifierToReadabilityTest(ReadableTestCase):
    def test_readable_draft_passes_through_without_extra_calls(self) -> None:
        """
        The Simplifier produces an easy draft; Readability scores it, sees it
        already meets the target, and returns it unchanged without calling Gemini.
        """

        print_scoring_rule()
        client = FakeGeminiClient(simplifier_replies(ATOMS, NO_TERMS, EASY_PLAIN))

        draft = simplifier.process_simplification(ATOMS, client, retriever=FakeRetriever())["draft"]
        result = readability.process_readability(draft, client)

        print_scores("Simplifier draft", draft, result["original_scores"])

        self.check("meets target", True, result["meets_target"])
        self.check("readability rewrites", 0, result["rewrite_attempts"])
        self.check("final text is the draft", draft, result["final_text"])
        self.check("total Gemini calls (Simplifier only)", 2, len(client.calls))

    def test_hard_draft_is_rewritten_by_readability(self) -> None:
        """
        The Simplifier produces a draft that is still too hard; Readability
        sends that exact draft to Gemini, gets an easier rewrite, and flags it
        for a faithfulness check because the text changed.
        """

        print_scoring_rule()
        client = FakeGeminiClient(
            simplifier_replies(ATOMS, NO_TERMS, HARD_PLAIN),
            plain_reply=READABLE_REWRITE,
        )

        draft = simplifier.process_simplification(ATOMS, client, retriever=FakeRetriever())["draft"]
        result = readability.process_readability(draft, client)

        print_scores("Simplifier draft", draft, result["original_scores"])
        print_scores("Readability final", result["final_text"], result["final_scores"])

        self.check("draft meets target", False, result["original_scores"]["meets_target"])
        self.check("final text", READABLE_REWRITE, result["final_text"])
        self.check("final meets target", True, result["meets_target"])
        self.check("needs faithfulness check", True, result["requires_faithfulness_check"])

        # Readability was asked to rewrite the Simplifier's exact draft.
        readability_call = client.calls[2]
        self.check("Readability call is plain text (not JSON)", False, readability_call["json_mode"])
        self.check_in("Readability prompt contains the draft", draft, readability_call["contents"])


class AdkSequentialTest(ReadableTestCase):
    def test_draft_flows_from_simplifier_to_readability(self) -> None:
        """
        Runs Simplifier -> Readability as a real ADK SequentialAgent: the
        Simplifier's output_key "draft" must reach the Readability Agent
        through session state, which then writes "final" and its scores.
        """

        print_scoring_rule()
        client = FakeGeminiClient(
            simplifier_replies(ATOMS, NO_TERMS, HARD_PLAIN),
            plain_reply=READABLE_REWRITE,
        )

        # SequentialAgent is deprecated in ADK 2.x but is what adk_pipeline.py uses.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pipeline = SequentialAgent(
                name="SimplifyThenRead",
                sub_agents=[simplifier.simplifier_agent, readability.readability_agent],
            )

        async def run() -> dict:
            sessions = InMemorySessionService()
            session = await sessions.create_session(
                app_name="test", user_id="u", state={"atoms": json.dumps(ATOMS)},
            )
            runner = Runner(app_name="test", agent=pipeline, session_service=sessions)

            async for _ in runner.run_async(
                user_id="u",
                session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text="go")]),
            ):
                pass

            done = await sessions.get_session(app_name="test", user_id="u", session_id=session.id)
            return done.state

        # Both callbacks build genai.Client(); patching it keeps the run offline.
        with patch("google.genai.Client", return_value=client), \
                patch.object(simplifier, "retrieve", FakeRetriever()):
            state = asyncio.run(run())

        print_scores("state['final']", state["final"], {
            **state["readability_scores"], "meets_target": state["readability_passed"],
        })

        self.check("state['draft'] from Simplifier", " ".join(HARD_PLAIN), state["draft"])
        self.check("state['simplified_atoms'] count", 3, len(state["simplified_atoms"]))
        self.check("state['final'] from Readability", READABLE_REWRITE, state["final"])
        self.check("state['readability_passed']", True, state["readability_passed"])
        self.check_in("Readability prompt contains state['draft']", state["draft"], client.calls[2]["contents"])


@unittest.skipUnless(RUN_LIVE, "set RUN_LIVE_GEMINI=1 to call Gemini")
class LiveSimplifierToReadabilityTest(ReadableTestCase):
    def test_two_atoms_through_both_agents(self) -> None:
        """Real Gemini runs 2 short atoms through both agents; the dose must survive both rewrites."""

        if not os.getenv("GEMINI_API_KEY"):
            self.skipTest("GEMINI_API_KEY is not set (put .env in the repo root)")

        from google import genai

        print_scoring_rule()
        atoms = ATOMS[:1] + ATOMS[2:]
        client = TokenCountingClient(genai.Client())

        try:
            draft = simplifier.process_simplification(atoms, client, retriever=live_retriever())["draft"]
            result = readability.process_readability(draft, client)
        finally:
            client.close()

        print(f"  Cost: {client.calls} Gemini calls, {client.total_tokens} tokens "
              f"({result['rewrite_attempts']} Readability rewrites)")
        print_scores("Simplifier draft", draft, result["original_scores"])
        print_scores("Readability final", result["final_text"], result["final_scores"])

        self.check_true(f"Gemini calls <= {2 + readability.MAX_REWRITES}",
                        client.calls <= 2 + readability.MAX_REWRITES)
        self.check_in("dose kept verbatim", "20 mg", result["final_text"])


def load_tests(loader, tests, pattern):
    return definition_order_suite(__name__)


if __name__ == "__main__":
    unittest.main(verbosity=0)
