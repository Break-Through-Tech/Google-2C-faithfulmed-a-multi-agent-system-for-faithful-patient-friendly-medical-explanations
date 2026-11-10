from __future__ import annotations

import asyncio
import unittest

from refiner.agent import RefinerConfig
from refiner.runner import RefinerRunner, refine
from refiner.prompts import build_case_message
from refiner.schemas import EvaluationFixture, RefinerInput
from refiner.validation import RefinerResponseError


class RefinerRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_flags_bypass_invoker_and_preserve_exactly(self) -> None:
        calls = 0

        def invoker(_case):
            nonlocal calls
            calls += 1
            raise AssertionError("must not be called")

        case = RefinerInput(source_text="Source.", draft="Draft  with\nspacing.")
        output = await RefinerRunner(model_invoker=invoker).refine_case(case)
        self.assertFalse(output.needs_refinement)
        self.assertEqual(output.refined_text, case.draft)
        self.assertEqual(calls, 0)

    async def test_malformed_response_is_explicit_error(self) -> None:
        case = RefinerInput.model_validate(
            {
                "source_text": "A 4 mm stone.",
                "draft": "A 9 mm stone.",
                "verifier_report": {
                    "faithful": False,
                    "unsupported_claims": ["Size is wrong."],
                    "omissions": [],
                    "reading_level_ok": True,
                },
            }
        )
        with self.assertRaises(RefinerResponseError):
            await RefinerRunner(model_invoker=lambda _case: "not json").refine_case(case)

    async def test_no_decision_cannot_modify_text(self) -> None:
        case = RefinerInput.model_validate(
            {
                "source_text": "A 4 mm stone.",
                "draft": "A 4 mm stone.",
                "verifier_report": {
                    "faithful": False,
                    "unsupported_claims": ["Size is wrong."],
                    "omissions": [],
                    "reading_level_ok": True,
                },
            }
        )
        response = {"needs_refinement": False, "refined_text": "Rewritten.", "unresolved_issues": []}
        with self.assertRaises(RefinerResponseError):
            await RefinerRunner(model_invoker=lambda _case: response).refine_case(case)

    async def test_async_model_call_has_total_deadline(self) -> None:
        case = RefinerInput.model_validate(
            {
                "source_text": "A 4 mm stone.",
                "draft": "A 9 mm stone.",
                "verifier_report": {
                    "faithful": False,
                    "unsupported_claims": ["Size is wrong."],
                },
            }
        )

        async def never_finishes(_case):
            await asyncio.sleep(1)
            return {}

        runner = RefinerRunner(
            config=RefinerConfig(request_timeout_seconds=0.01),
            model_invoker=never_finishes,
        )
        with self.assertRaisesRegex(RefinerResponseError, "within 0.01 seconds"):
            await runner.refine_case(case)


class PublicInterfaceTests(unittest.TestCase):
    def test_sync_refine(self) -> None:
        output = refine("Source.", "Draft.", None)
        self.assertEqual(output.refined_text, "Draft.")

    def test_gold_fixture_fields_are_not_serialized_for_model(self) -> None:
        fixture = EvaluationFixture(
            source_text="Source fact.",
            draft="Draft fact.",
            verifier_report={"unsupported_claims": ["Flag."], "faithful": False},
            gold_needs_refinement=True,
            category="test",
            gold_flagged_issues=["Flag."],
            expected_corrections=["SECRET_EXPECTED_CORRECTION"],
            gold_refined_text="SECRET_GOLD_TEXT",
            preserve_facts=[],
            expected_unresolved_issues=[],
            annotation_provenance="test",
        )
        message = build_case_message(fixture)
        self.assertNotIn("SECRET_EXPECTED_CORRECTION", message)
        self.assertNotIn("SECRET_GOLD_TEXT", message)


if __name__ == "__main__":
    unittest.main()
