"""CLI demonstration for independent Refiner cases."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from refiner import RefinerConfig, RefinerInput, RefinerOutput, RefinerRunner
from refiner.evaluation import load_fixtures, select_diverse_fixtures

MAX_LIVE_CASES = 20
DEFAULT_LIVE_REQUEST_INTERVAL_SECONDS = 4.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("tests/fixtures/urology_dev.jsonl"))
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="number of category-balanced cases (default: 5; live maximum: 20)",
    )
    parser.add_argument(
        "--mode",
        choices=("live", "fixture-mock"),
        default="live",
        help="fixture-mock demonstrates plumbing only and never calls Gemini",
    )
    parser.add_argument(
        "--verbose-adk",
        action="store_true",
        help="show internal ADK tracebacks instead of concise operational errors",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="total deadline for each model call, including retries (default: 45)",
    )
    parser.add_argument(
        "--output-format",
        choices=("human", "jsonl"),
        default="human",
        help="human-readable details or one exact RefinerOutput JSON object per line",
    )
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=DEFAULT_LIVE_REQUEST_INTERVAL_SECONDS,
        help="minimum delay between live Gemini calls (default: 4.1; ignored in fixture-mock)",
    )
    return parser.parse_args()


async def refine_with_progress(
    runner: RefinerRunner,
    fixture,
    *,
    status_stream: TextIO = sys.stdout,
) -> RefinerOutput:
    """Keep a live CLI visibly responsive while the network request is pending."""

    task = asyncio.create_task(runner.refine_case(fixture))
    started = asyncio.get_running_loop().time()
    while not task.done():
        done, _ = await asyncio.wait({task}, timeout=5.0)
        if not done:
            elapsed = asyncio.get_running_loop().time() - started
            print(
                f"Waiting for Gemini... {elapsed:.0f}s elapsed",
                file=status_stream,
                flush=True,
            )
    return await task


async def main_async(args: argparse.Namespace) -> int:
    jsonl = args.output_format == "jsonl"
    status_stream = sys.stderr if jsonl else sys.stdout

    def status(message: str = "") -> None:
        print(message, file=status_stream, flush=True)

    if not args.verbose_adk:
        # ADK 2.8 uses a ``google_adk.google.adk.*`` logger hierarchy internally.
        # Keep retry tracebacks out of normal CLI output; the final operational
        # error is still caught and printed below.
        logging.getLogger("google_adk").setLevel(logging.CRITICAL)
        logging.getLogger("google.adk.workflow._node_runner").setLevel(logging.CRITICAL)
        logging.getLogger("google.adk.runners").setLevel(logging.CRITICAL)
    if args.mode == "live" and args.limit > MAX_LIVE_CASES:
        raise SystemExit(f"live demos are limited to {MAX_LIVE_CASES} cases")
    request_interval = getattr(
        args, "request_interval_seconds", DEFAULT_LIVE_REQUEST_INTERVAL_SECONDS
    )
    if request_interval < 0:
        raise SystemExit("--request-interval-seconds cannot be negative")
    fixtures = select_diverse_fixtures(load_fixtures(args.input), args.limit)
    lookup = {fixture.example_id: fixture for fixture in fixtures}

    def fixture_mock(case) -> dict:
        fixture = lookup[case.example_id]
        return RefinerOutput(
            needs_refinement=fixture.gold_needs_refinement,
            refined_text=fixture.gold_refined_text,
            unresolved_issues=fixture.expected_unresolved_issues,
        ).model_dump()

    config = RefinerConfig.from_env()
    if args.timeout_seconds is not None:
        config = replace(config, request_timeout_seconds=args.timeout_seconds)
    if config.request_timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be greater than zero")
    runner = RefinerRunner(
        config=config,
        model_invoker=fixture_mock if args.mode == "fixture-mock" else None,
    )
    status(f"Run mode: {args.mode}")
    if args.mode == "fixture-mock":
        status("NOTICE: outputs are fixture-backed and are not Gemini performance results.")
    else:
        status(f"Model: {runner.config.model}")
        status(
            "Per-case deadline: "
            f"{runner.config.request_timeout_seconds:g}s, including transient-error retries."
        )
    errors = 0
    last_live_request_started: float | None = None
    for index, fixture in enumerate(fixtures, start=1):
        runtime_input = RefinerInput(
            example_id=fixture.example_id,
            source_text=fixture.source_text,
            draft=fixture.draft,
            verifier_report=fixture.verifier_report,
        )
        if not jsonl:
            status("\n" + "=" * 78)
            status(f"TEST CASE {index}/{len(fixtures)} — {fixture.example_id}")
            status("=" * 78)
            status("INPUT JSON (exact RefinerInput):")
            status(runtime_input.model_dump_json(indent=2))
        else:
            status(f"Case ID: {fixture.example_id}")
        try:
            if args.mode == "live" and fixture.verifier_report.flagged_issues:
                if last_live_request_started is not None:
                    elapsed = time.monotonic() - last_live_request_started
                    wait_seconds = max(0.0, request_interval - elapsed)
                    if wait_seconds > 0:
                        status(
                            f"Rate-limit pacing: waiting {wait_seconds:.1f}s before Gemini call..."
                        )
                        await asyncio.sleep(wait_seconds)
                last_live_request_started = time.monotonic()
                status("Calling Gemini...")
                output = await refine_with_progress(
                    runner,
                    fixture,
                    status_stream=status_stream,
                )
            else:
                output = await runner.refine_case(fixture)
        except Exception as exc:
            if not jsonl:
                status("OUTPUT ERROR JSON (no RefinerOutput was produced):")
                status(
                    json.dumps(
                        {"error_type": type(exc).__name__, "message": str(exc)},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                status("-" * 78)
                status(f"END TEST CASE {index}/{len(fixtures)} — {fixture.example_id}")
            else:
                status(f"Operational error: {type(exc).__name__}: {exc}")
            errors += 1
            continue
        if jsonl:
            # stdout stays machine-readable: one exact RefinerOutput object per case.
            print(output.model_dump_json(), flush=True)
        else:
            status("OUTPUT JSON (exact RefinerOutput):")
            status(output.model_dump_json(indent=2))
            status(f"Execution time: {runner.last_latency_seconds:.4f} seconds")
            status("-" * 78)
            status(f"END TEST CASE {index}/{len(fixtures)} — {fixture.example_id}")
    return 1 if errors else 0


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
