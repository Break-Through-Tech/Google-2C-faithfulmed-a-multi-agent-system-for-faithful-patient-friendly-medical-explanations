from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from refiner.agent import REFINER_OUTPUT_SCHEMA, create_refiner_agent
from refiner.schemas import RefinerInput, RefinerOutput, VerifierReport


class RefinerSchemaTests(unittest.TestCase):
    def test_defaults_missing_report_to_no_flags(self) -> None:
        case = RefinerInput(source_text="A source.", draft="A draft.")
        self.assertTrue(case.verifier_report.faithful)
        self.assertEqual(case.verifier_report.flagged_issues, [])

    def test_empty_source_is_invalid(self) -> None:
        with self.assertRaises(ValidationError):
            RefinerInput(source_text="  ", draft="A draft.")

    def test_unknown_verifier_field_is_invalid(self) -> None:
        with self.assertRaises(ValidationError):
            VerifierReport.model_validate({"faithful": True, "unexpected": 1})

    def test_unresolved_requires_yes(self) -> None:
        with self.assertRaises(ValidationError):
            RefinerOutput(
                needs_refinement=False,
                refined_text="Unchanged.",
                unresolved_issues=["Ambiguous."],
            )

    def test_json_round_trip(self) -> None:
        output = RefinerOutput(needs_refinement=True, refined_text="Fixed.", unresolved_issues=[])
        self.assertEqual(RefinerOutput.model_validate_json(output.model_dump_json()), output)
        self.assertEqual(json.loads(output.model_dump_json())["refined_text"], "Fixed.")

    def test_adk_schema_omits_unsupported_additional_properties(self) -> None:
        schema = REFINER_OUTPUT_SCHEMA.model_dump(by_alias=True, exclude_none=True)
        self.assertNotIn("additionalProperties", schema)
        self.assertNotIn("additional_properties", schema)
        self.assertEqual(
            set(schema["required"]),
            {"needs_refinement", "refined_text", "unresolved_issues"},
        )

    def test_agent_uses_native_json_without_afc(self) -> None:
        agent = create_refiner_agent()
        self.assertIs(agent.output_schema, REFINER_OUTPUT_SCHEMA)
        config = agent.generate_content_config
        self.assertIsNone(config.response_schema)
        self.assertTrue(config.automatic_function_calling.disable)
        self.assertEqual(agent.retry_config.max_attempts, 3)
        self.assertEqual(agent.retry_config.exceptions, ["ServerError"])


if __name__ == "__main__":
    unittest.main()
