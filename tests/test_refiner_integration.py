from __future__ import annotations

import unittest
from types import SimpleNamespace

from refiner.integration import (
    build_pipeline_instruction,
    create_pipeline_refiner_agent,
    pipeline_case_from_state,
)
from refiner.validation import RefinerConfigurationError


class RefinerPipelineIntegrationTests(unittest.TestCase):
    def test_agent_matches_repository_state_shape(self) -> None:
        agent = create_pipeline_refiner_agent()
        self.assertEqual(agent.name, "Refiner")
        self.assertEqual(agent.output_key, "draft")
        self.assertIsNone(agent.output_schema)
        self.assertEqual(agent.include_contents, "none")

    def test_pipeline_state_maps_to_standalone_contract(self) -> None:
        case = pipeline_case_from_state(
            {
                "source_text": "CT shows a 3 mm left ureter stone.",
                "draft": "Your CT shows an 8 mm left ureter stone.",
                "verdict": {
                    "faithful": False,
                    "unsupported_claims": ["The stone measurement is incorrect."],
                    "omissions": [],
                    "reading_level_ok": True,
                },
            }
        )
        self.assertEqual(case.source_text, "CT shows a 3 mm left ureter stone.")
        self.assertEqual(case.verifier_report.flagged_issues, ["The stone measurement is incorrect."])

    def test_json_verdict_from_scaffold_is_accepted(self) -> None:
        case = pipeline_case_from_state(
            {
                "source_text": "No fever.",
                "draft": "You have a fever.",
                "verdict": '{"faithful": false, "unsupported_claims": ["Fever is wrong."], '
                '"omissions": [], "reading_level_ok": true}',
            }
        )
        self.assertFalse(case.verifier_report.faithful)

    def test_instruction_uses_latest_shared_state(self) -> None:
        context = SimpleNamespace(
            state={
                "source_text": "Source fact.",
                "draft": "Draft fact.",
                "verdict": {
                    "faithful": False,
                    "unsupported_claims": ["Flagged fact."],
                    "omissions": [],
                    "reading_level_ok": True,
                },
            }
        )
        instruction = build_pipeline_instruction(context)
        self.assertIn("Source fact.", instruction)
        self.assertIn("Draft fact.", instruction)
        self.assertIn("Flagged fact.", instruction)
        self.assertIn("replaces the shared `draft` state", instruction)

    def test_missing_source_fails_before_model_call(self) -> None:
        with self.assertRaisesRegex(RefinerConfigurationError, "missing: source_text"):
            pipeline_case_from_state({"draft": "Draft.", "verdict": {}})


if __name__ == "__main__":
    unittest.main()
