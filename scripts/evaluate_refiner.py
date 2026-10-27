"""Evaluate live Gemini or the offline fixture-backed contract harness."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from refiner import RefinerOutput, RefinerRunner
from refiner.evaluation import (
    aggregate_metrics,
    build_report_metadata,
    load_fixtures,
    readability_scores,
    score_prediction,
    select_diverse_fixtures,
    write_artifacts,
)
from refiner.schemas import PredictionRecord

MAX_LIVE_CASES = 20
DEFAULT_LIVE_REQUEST_INTERVAL_SECONDS = 4.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("tests/fixtures/urology_dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/refiner_dev"))
    parser.add_argument(
        "--limit",
        type=int,
        help="number of category-balanced cases (live default and maximum: 20)",
    )
    parser.add_argument("--mode", choices=("live", "fixture-mock"), default="fixture-mock")
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=DEFAULT_LIVE_REQUEST_INTERVAL_SECONDS,
        help="minimum delay between live Gemini calls (default: 4.1; ignored in fixture-mock)",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    fixtures = load_fixtures(args.input)
    requested_limit = args.limit
    if args.mode == "live":
        requested_limit = MAX_LIVE_CASES if requested_limit is None else requested_limit
        if requested_limit > MAX_LIVE_CASES:
            raise SystemExit(f"live evaluations are limited to {MAX_LIVE_CASES} cases")
    if requested_limit is not None and requested_limit < 1:
        raise SystemExit("--limit must be at least 1")
    fixtures = select_diverse_fixtures(fixtures, requested_limit)
    request_interval = getattr(
        args, "request_interval_seconds", DEFAULT_LIVE_REQUEST_INTERVAL_SECONDS
    )
    if request_interval < 0:
        raise SystemExit("--request-interval-seconds cannot be negative")
    lookup = {fixture.example_id: fixture for fixture in fixtures}

    def fixture_mock(case) -> dict:
        fixture = lookup[case.example_id]
        return RefinerOutput(
            needs_refinement=fixture.gold_needs_refinement,
            refined_text=fixture.gold_refined_text,
            unresolved_issues=fixture.expected_unresolved_issues,
        ).model_dump()

    run_type = "live-gemini" if args.mode == "live" else "fixture-mock"
    runner = RefinerRunner(model_invoker=None if args.mode == "live" else fixture_mock)
    records: list[PredictionRecord] = []
    last_live_request_started: float | None = None
    for fixture in fixtures:
        try:
            if args.mode == "live" and fixture.verifier_report.flagged_issues:
                if last_live_request_started is not None:
                    elapsed = time.monotonic() - last_live_request_started
                    await asyncio.sleep(max(0.0, request_interval - elapsed))
                last_live_request_started = time.monotonic()
            output = await runner.refine_case(fixture)
            record = score_prediction(
                fixture,
                output,
                runner.last_latency_seconds or 0.0,
                runner.last_usage_metadata,
            )
        except Exception as exc:
            record = PredictionRecord(
                example_id=fixture.example_id,
                category=fixture.category,
                source_text=fixture.source_text,
                draft=fixture.draft,
                verifier_report=fixture.verifier_report,
                gold_needs_refinement=fixture.gold_needs_refinement,
                gold_refined_text=fixture.gold_refined_text,
                error=f"{type(exc).__name__}: {exc}",
                latency_seconds=runner.last_latency_seconds or 0.0,
                readability_before=readability_scores(fixture.draft),
                readability_after={},
            )
        records.append(record)
    metrics = aggregate_metrics(records)
    metadata = build_report_metadata(
        dataset=str(args.input),
        run_type=run_type,
        config=runner.config,
        count=len(fixtures),
    )
    write_artifacts(args.output_dir, metadata, metrics, records)
    print(f"Wrote evaluation artifacts to {args.output_dir}")
    print(f"Run type: {run_type}")
    print(f"Examples: {len(fixtures)}; errors: {metrics['errors']}")
    print(f"Binary F1: {metrics['classification']['f1']:.3f}")
    if run_type == "fixture-mock":
        print("NOTICE: fixture-mock results validate code, not Gemini quality.")
    return 1 if metrics["errors"] else 0


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
