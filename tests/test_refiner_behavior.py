from __future__ import annotations

import unittest
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

from refiner.evaluation import load_fixtures, readability_scores
from refiner.runner import RefinerRunner

FIXTURES = Path(__file__).parent / "fixtures" / "urology_dev.jsonl"


class RefinerBehaviorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.fixtures = load_fixtures(FIXTURES)
        self.lookup = {fixture.example_id: fixture for fixture in self.fixtures}

        def mock(case):
            fixture = self.lookup[case.example_id]
            return {
                "needs_refinement": fixture.gold_needs_refinement,
                "refined_text": fixture.gold_refined_text,
                "unresolved_issues": fixture.expected_unresolved_issues,
            }

        self.runner = RefinerRunner(model_invoker=mock)

    async def test_fixture_count_and_distribution(self) -> None:
        self.assertEqual(len(self.fixtures), 50)
        counts: dict[str, int] = {}
        for fixture in self.fixtures:
            counts[fixture.category] = counts.get(fixture.category, 0) + 1
        self.assertEqual(set(counts.values()), {10})

    async def test_valid_measurement_correction(self) -> None:
        fixture = self.lookup["uro_dev_001"]
        output = await self.runner.refine_case(fixture)
        self.assertTrue(output.needs_refinement)
        self.assertIn("3 mm stone", output.refined_text)
        self.assertIn("you do not have a fever.", output.refined_text)
        self.assertIn("ureterovesical junction", fixture.source_text)
        self.assertNotIn("ureterovesical junction", fixture.draft)
        self.assertNotIn("calculus", output.refined_text)
        self.assertNotIn("hydroureteronephrosis", output.refined_text)

    async def test_multiple_flags(self) -> None:
        fixture = self.lookup["uro_dev_005"]
        output = await self.runner.refine_case(fixture)
        self.assertIn("4 mm stone near the bottom of your left ureter", output.refined_text)

    async def test_sources_are_clinical_and_drafts_are_lay_language(self) -> None:
        for split in ("dev", "test"):
            with self.subTest(split=split):
                fixtures = load_fixtures(
                    Path(__file__).parent / "fixtures" / f"urology_{split}.jsonl"
                )
                source_grades = [
                    float(readability_scores(fixture.source_text)["flesch_kincaid_grade"])
                    for fixture in fixtures
                ]
                draft_grades = [
                    float(readability_scores(fixture.draft)["flesch_kincaid_grade"])
                    for fixture in fixtures
                ]
                similarities = [
                    SequenceMatcher(
                        None,
                        fixture.source_text.lower(),
                        fixture.draft.lower(),
                    ).ratio()
                    for fixture in fixtures
                ]

                self.assertGreater(mean(source_grades) - mean(draft_grades), 5.0)
                self.assertTrue(
                    all(source > draft for source, draft in zip(source_grades, draft_grades))
                )
                self.assertLess(mean(similarities), 0.55)
                self.assertLess(max(similarities), 0.65)

    async def test_refinement_preserves_simplifiers_lay_wording(self) -> None:
        for split in ("dev", "test"):
            with self.subTest(split=split):
                fixtures = load_fixtures(
                    Path(__file__).parent / "fixtures" / f"urology_{split}.jsonl"
                )
                similarities = []
                for fixture in fixtures:
                    if fixture.gold_needs_refinement and not fixture.expected_unresolved_issues:
                        similarities.append(
                            SequenceMatcher(
                                None,
                                fixture.draft.lower(),
                                fixture.gold_refined_text.lower(),
                            ).ratio()
                        )
                self.assertGreater(mean(similarities), 0.85)

    async def test_unresolved_flag_preserves_draft(self) -> None:
        fixture = self.lookup["uro_dev_030"]
        output = await self.runner.refine_case(fixture)
        self.assertTrue(output.needs_refinement)
        self.assertEqual(output.refined_text, fixture.draft)
        self.assertEqual(output.unresolved_issues, fixture.expected_unresolved_issues)

    async def test_false_flag_is_rejected(self) -> None:
        fixture = self.lookup["uro_dev_041"]
        output = await self.runner.refine_case(fixture)
        self.assertFalse(output.needs_refinement)
        self.assertEqual(output.refined_text, fixture.draft)

    async def test_missed_error_is_not_corrected(self) -> None:
        fixture = self.lookup["uro_dev_046"]
        output = await self.runner.refine_case(fixture)
        self.assertFalse(output.needs_refinement)
        self.assertEqual(output.refined_text, fixture.draft)
        self.assertNotEqual(output.refined_text, fixture.source_text)


if __name__ == "__main__":
    unittest.main()
