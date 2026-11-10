from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from refiner.agent import RefinerConfig
from refiner.evaluation import (
    aggregate_metrics,
    build_report_metadata,
    load_fixtures,
    score_prediction,
    select_diverse_fixtures,
    write_artifacts,
)
from refiner.schemas import RefinerOutput


class EvaluationTests(unittest.TestCase):
    def test_report_defines_evaluation_vocabulary_inline(self) -> None:
        fixture = load_fixtures(Path(__file__).parent / "fixtures" / "urology_dev.jsonl")[0]
        output = RefinerOutput(
            needs_refinement=fixture.gold_needs_refinement,
            refined_text=fixture.gold_refined_text,
            unresolved_issues=fixture.expected_unresolved_issues,
        )
        records = [score_prediction(fixture, output, 0.01)]
        metrics = aggregate_metrics(records)
        metadata = build_report_metadata(
            dataset="fixture.jsonl",
            run_type="fixture-mock",
            config=RefinerConfig(),
            count=1,
        )

        with TemporaryDirectory() as directory:
            write_artifacts(directory, metadata, metrics, records)
            report = (Path(directory) / "evaluation_report.md").read_text(encoding="utf-8")

        self.assertIn("**Binary decision**", report)
        self.assertIn("Small definition", report)
        self.assertIn("**False-flag rejection rate**", report)
        self.assertIn("**gold** answer", report)
        self.assertIn("Successfully scored: 1", report)

    def test_diverse_limit_balances_and_interleaves_categories(self) -> None:
        fixtures = load_fixtures(Path(__file__).parent / "fixtures" / "urology_dev.jsonl")

        selected = select_diverse_fixtures(fixtures, 20)

        self.assertEqual(len(selected), 20)
        category_counts = {
            category: sum(fixture.category == category for fixture in selected)
            for category in {fixture.category for fixture in fixtures}
        }
        self.assertEqual(set(category_counts.values()), {4})
        self.assertEqual(len({fixture.category for fixture in selected[:5]}), 5)
        self.assertEqual(
            [fixture.example_id for fixture in selected[:5]],
            ["uro_dev_001", "uro_dev_011", "uro_dev_021", "uro_dev_031", "uro_dev_041"],
        )
        self.assertIn("uro_dev_050", {fixture.example_id for fixture in selected})

    def test_fixture_mock_metrics_and_formulas(self) -> None:
        fixtures = load_fixtures(Path(__file__).parent / "fixtures" / "urology_dev.jsonl")
        records = []
        for fixture in fixtures:
            output = RefinerOutput(
                needs_refinement=fixture.gold_needs_refinement,
                refined_text=fixture.gold_refined_text,
                unresolved_issues=fixture.expected_unresolved_issues,
            )
            records.append(score_prediction(fixture, output, 0.01))
        metrics = aggregate_metrics(records)
        self.assertEqual(metrics["classification"]["accuracy"], 1.0)
        self.assertEqual(metrics["classification"]["f1"], 1.0)
        self.assertEqual(metrics["classification"]["always_yes_baseline_accuracy"], 0.8)
        self.assertEqual(metrics["correction_quality"]["flagged_issue_correction_rate"], 1.0)
        self.assertEqual(metrics["correction_quality"]["unflagged_edit_rate"], 0.0)
        self.assertIsNone(metrics["correction_quality"]["new_clinically_meaningful_error_rate"])


if __name__ == "__main__":
    unittest.main()
