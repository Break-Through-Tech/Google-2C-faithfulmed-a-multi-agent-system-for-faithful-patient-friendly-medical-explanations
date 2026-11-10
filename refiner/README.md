# FaithfulMed Urology Refiner

This package is a standalone Google ADK prototype for one narrow job: validate issues explicitly
flagged by the Verifier against the original source, then make only the smallest supported changes.
It does not run the other FaithfulMed agents and does not modify `notebooks/adk_pipeline.py`.

## Architecture

```text
source + draft + verifier report
             |
             v
   deterministic no-flag gate ------> exact unchanged output
             |
      explicit flags only
             v
 Google ADK LlmAgent (Gemini, structured RefinerOutput)
             |
             v
 Pydantic parsing + invariant validation
             |
             v
 needs_refinement / refined_text / unresolved_issues
```

The no-flag gate prevents a model call and preserves the draft exactly. `faithful=false` or
`reading_level_ok=false` alone does not authorize edits. For flagged cases, the versioned prompt
treats source/draft/report text as untrusted data and requires source-grounded, flag-scoped edits.
The validator rejects malformed JSON, extra fields, a changed draft paired with a NO decision, an
empty YES decision, and unresolved issues that do not exactly match a supplied flag. Semantic
medical correctness still requires model evaluation and independent clinical review; deterministic
code cannot prove it from free text.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
```

The model defaults to `gemini-3.5-flash-lite` and is replaceable. The repository's original
`gemini-2.0-flash` starter model has been shut down:

```bash
export FAITHFULMED_REFINER_MODEL="gemini-3.5-flash-lite"
export FAITHFULMED_REFINER_TEMPERATURE="0"
export FAITHFULMED_REFINER_MAX_OUTPUT_TOKENS="2048"
export FAITHFULMED_REFINER_TIMEOUT_SECONDS="45"
```

The implementation was validated locally with Python 3.13, `google-adk==2.8.0`,
`google-genai==2.22.0`, and `pydantic==2.13.4`. Google model availability changes over time; choose
an available Gemini model through `FAITHFULMED_REFINER_MODEL` rather than editing code.

## Use

Python API:

```python
from refiner import refine

result = refine(
    source_text="CT shows a 4 mm stone in the left ureter.",
    draft="Your CT shows a 9 mm stone in the left ureter.",
    verifier_report={
        "faithful": False,
        "unsupported_claims": ["The stone measurement is incorrect."],
        "omissions": [],
        "reading_level_ok": True,
    },
)
print(result.model_dump())
```

Use `await refine_async(...)` in notebooks or other running event loops.

Live CLI demo:

```bash
python -m scripts.run_refiner_demo \
  --input tests/fixtures/urology_dev.jsonl \
  --limit 20 \
  --timeout-seconds 45 \
  --mode live
```

Live commands accept at most 20 cases. Limited runs use a deterministic, category-balanced sample
instead of taking the first rows: a 20-case run includes four examples from each of the five fixture
categories and interleaves those categories. Live Gemini calls are spaced 4.1 seconds apart by
default to stay within the observed 15-requests-per-minute free-tier quota. Override the interval
only when the API project's quota supports it with `--request-interval-seconds`.

The default terminal format clearly separates each case and prints two labeled JSON blocks:
the exact runtime `RefinerInput` and the exact validated `RefinerOutput`. Evaluation-only gold
fields are never included in the input block.

Print only the exact `RefinerOutput` contract as JSON Lines (operational status goes to stderr):

```bash
python -m scripts.run_refiner_demo \
  --input tests/fixtures/urology_dev.jsonl \
  --limit 5 \
  --mode live \
  --timeout-seconds 45 \
  --output-format jsonl
```

Each stdout line has exactly these fields:

```json
{"needs_refinement":true,"refined_text":"...","unresolved_issues":[]}
```

The timeout is an end-to-end per-case deadline, including ADK retries. While a live request is
pending, the CLI prints an elapsed-time update every five seconds.

Offline wiring demo (gold fixture responses, no Gemini call):

```bash
python -m scripts.run_refiner_demo --limit 5 --mode fixture-mock
```

The offline mode is only a contract/harness demonstration. Its outputs are not model results.

## Tests and evaluation

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
  -m unittest discover -s tests -v

# Refiner-to-pipeline state-contract tests only
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
  -m unittest discover -s tests -p 'test_refiner_integration.py' -v

# Optional one-case live Gemini test; prints the exact RefinerOutput JSON
RUN_LIVE_GEMINI=1 \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
  -m unittest discover -s tests -p 'test_refiner_live.py' -v

# Offline harness validation
python -m scripts.evaluate_refiner \
  --input tests/fixtures/urology_dev.jsonl \
  --output-dir reports/refiner_dev_mock \
  --mode fixture-mock

# Actual development-set model evaluation
python -m scripts.evaluate_refiner \
  --input tests/fixtures/urology_dev.jsonl \
  --output-dir reports/refiner_dev_live_20 \
  --limit 20 \
  --mode live
```

If `--limit` is omitted for a live evaluation, it defaults to 20. Fixture-mock evaluations still
process all 50 cases unless a limit is supplied. Report schemas and terminal JSON formats are the
same for full and limited runs; only the number and balanced selection of examples changes.

Each evaluation writes `metrics.json`, `evaluation_report.md`, and `predictions.jsonl`. Live runs
are labeled `live-gemini`; fixture-backed runs are labeled `fixture-mock`. The generated synthetic
annotations are provisional and not independently clinically reviewed. Consequently, clinical
faithfulness, omission, and new-error metrics remain explicitly pending rather than being inferred
from the Refiner's own decisions.

`textstat` is the primary FK/SMOG backend. If its optional NLTK `cmudict` resource is unavailable,
the evaluator uses a documented deterministic syllable-count fallback so offline tests do not
download data.

## Fixture policy

`tests/fixtures/urology_dev.jsonl` and `urology_test.jsonl` each contain 50 synthetic cases. Do not
tune the prompt on the test split. The test split should be treated as frozen. The generator is
retained for provenance; running it intentionally creates dataset version `urology-synthetic-v2`.

Version 2 models the actual pipeline boundary: source documents use synthetic clinical terminology,
Simplifier drafts use lay language, and gold Refiner outputs preserve that lay wording while making
only the verifier-authorized correction. Regression tests enforce a substantial readability gap
between sources and drafts and high textual similarity between drafts and refined outputs.

Gold fields are parsed only by evaluation code. `build_case_message()` serializes only source text,
draft text, example ID, and the Verifier report, preventing expected answers from reaching Gemini.

## ADK pipeline integration

Use `create_pipeline_refiner_agent()` inside the repository's bounded loop. It adapts the richer
standalone contract to the pipeline's shared-state contract:

| Pipeline state | Standalone field |
|---|---|
| original source document | `source_text` |
| current simplifier/refiner draft | `draft` |
| Verifier structured verdict | `verdict` (dict or JSON text) |
| corrected draft | `draft` (overwritten with plain text) |

The session must seed `source_text` with the original document before the sequence starts. The
adapter fails explicitly if `source_text`, `draft`, or `verdict` is absent. Its `output_key="draft"`
means the next Verifier iteration and final Readability agent receive exactly the corrected string,
not the standalone metadata object.

The orchestration layer owns `LoopAgent(max_iterations=2)`, re-verification, and loop completion.
The final Readability agent remains responsible for unflagged reading-level edits.
