from __future__ import annotations

import os
import unittest
from pathlib import Path

from refiner.evaluation import load_fixtures
from refiner.runner import RefinerRunner
from refiner.schemas import RefinerInput


@unittest.skipUnless(
    os.getenv("RUN_LIVE_GEMINI") == "1",
    "set RUN_LIVE_GEMINI=1 and GEMINI_API_KEY to run live Gemini tests",
)
class LiveGeminiTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_development_case(self) -> None:
        fixture = load_fixtures(Path(__file__).parent / "fixtures" / "urology_dev.jsonl")[0]
        runtime_input = RefinerInput(
            example_id=fixture.example_id,
            source_text=fixture.source_text,
            draft=fixture.draft,
            verifier_report=fixture.verifier_report,
        )
        print("\n" + "=" * 78)
        print(f"LIVE TEST CASE — {fixture.example_id}")
        print("=" * 78)
        print("INPUT JSON (exact RefinerInput):")
        print(runtime_input.model_dump_json(indent=2))
        output = await RefinerRunner().refine_case(fixture)
        print("OUTPUT JSON (exact RefinerOutput):")
        print(output.model_dump_json(indent=2))
        print("-" * 78)
        print(f"END LIVE TEST CASE — {fixture.example_id}")
        self.assertTrue(output.needs_refinement)
        self.assertIn("3 mm", output.refined_text)


if __name__ == "__main__":
    unittest.main()
