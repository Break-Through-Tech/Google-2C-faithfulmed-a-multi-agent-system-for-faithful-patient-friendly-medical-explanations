from __future__ import annotations

import argparse
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from scripts.run_refiner_demo import main_async


class RefinerCliTests(unittest.IsolatedAsyncioTestCase):
    def make_args(self, output_format: str) -> argparse.Namespace:
        return argparse.Namespace(
            input=Path("tests/fixtures/urology_dev.jsonl"),
            limit=1,
            mode="fixture-mock",
            verbose_adk=False,
            timeout_seconds=45.0,
            output_format=output_format,
            request_interval_seconds=0.0,
        )

    async def test_jsonl_stdout_is_exact_refiner_output_contract(self) -> None:
        args = self.make_args("jsonl")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = await main_async(args)

        self.assertEqual(exit_code, 0)
        lines = stdout.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(
            set(payload),
            {"needs_refinement", "refined_text", "unresolved_issues"},
        )
        self.assertIn("Run mode: fixture-mock", stderr.getvalue())

    async def test_human_output_separates_exact_input_and_output_json(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = await main_async(self.make_args("human"))

        rendered = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("TEST CASE 1/1 — uro_dev_001", rendered)
        self.assertIn("INPUT JSON (exact RefinerInput):", rendered)
        self.assertIn("OUTPUT JSON (exact RefinerOutput):", rendered)
        self.assertIn("END TEST CASE 1/1 — uro_dev_001", rendered)
        self.assertNotIn("gold_refined_text", rendered)
        self.assertLess(rendered.index("INPUT JSON"), rendered.index("OUTPUT JSON"))


if __name__ == "__main__":
    unittest.main()
