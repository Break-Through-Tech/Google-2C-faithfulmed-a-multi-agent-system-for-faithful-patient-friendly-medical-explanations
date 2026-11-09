"""
Run the Extractor on MTSamples clinical reports (no gold nuggets: qualitative check).

    .venv/bin/python -m extractor.run_mtsamples --pick            # show the seeded urology draw
    .venv/bin/python -m extractor.run_mtsamples 38 23 96          # run those rows (1 API call each)

Rows are the CSV's index column. Data: data/raw/mtsamples/mtsamples.csv (gitignored;
copy of Kaggle tboyle10/medicaltranscriptions). Results go to
extractor/runs/<prompt_version>/mtsamples_<row>_<timestamp>.json; view with show_run.
"""
from __future__ import annotations

import collections
import csv
import json
import logging
import random
import sys
from pathlib import Path

from extractor.agent import extract

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "mtsamples" / "mtsamples.csv"
RUNS = Path(__file__).with_name("runs")
MAX_REPORTS_PER_RUN = 5  # shared rate limits: ask before running more
SPECIALTY = "Urology"
PICK_SEED, PICK_N, MIN_CHARS, MAX_CHARS = 42, 3, 800, 2500


def load_rows() -> dict[str, dict]:
    with open(CSV, encoding="utf-8") as f:
        return {r[""]: r for r in csv.DictReader(f)}


def seeded_pick(rows: dict[str, dict]) -> list[dict]:
    pool = [r for r in rows.values()
            if r["medical_specialty"].strip() == SPECIALTY
            and MIN_CHARS <= len(r["transcription"]) <= MAX_CHARS]
    return random.Random(PICK_SEED).sample(pool, PICK_N)


def run(row: dict) -> dict:
    result = extract(f"mtsamples-{row['']}", row["transcription"])
    record = {
        "result": result.model_dump(),
        "source": {"dataset": "mtsamples", "row": row[""],
                   "specialty": row["medical_specialty"].strip(),
                   "sample_name": row["sample_name"].strip()},
        "scores": {
            "n_atoms": len(result.atoms),
            "evidence_failures": len(result.evidence_failures),
            "hedge_flags": len(result.hedge_flags),
            "types": dict(collections.Counter(a.type for a in result.atoms)),
            "hedges": dict(collections.Counter(a.hedge for a in result.atoms)),
        },
    }
    out_dir = RUNS / result.provenance.prompt_version
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.provenance.timestamp.replace(":", "").split(".")[0]
    (out_dir / f"mtsamples-{row['']}_{stamp}.json").write_text(json.dumps(record, indent=2))
    return record


def main(args: list[str]) -> None:
    rows = load_rows()
    if args == ["--pick"]:
        for r in seeded_pick(rows):
            print(f"{r['']:>5}  {r['sample_name'].strip()}  ({len(r['transcription'])} chars)")
        return
    if len(args) > MAX_REPORTS_PER_RUN:
        sys.exit(f"{len(args)} reports > {MAX_REPORTS_PER_RUN}; ask before larger runs.")
    for rid in args:
        if rid not in rows:
            print(f"{rid}: no such row")
            continue
        try:
            rec = run(rows[rid])
        except Exception as e:  # report and move on; never retry here
            print(f"{rid}: FAILED {type(e).__name__}: {str(e)[:200]}")
            continue
        print(f"{rid} ({rec['source']['sample_name']}): {json.dumps(rec['scores'])}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    log = logging.getLogger("faithfulmed.extractor")
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())
    log.propagate = False
    main(sys.argv[1:])
