"""
Run the Extractor on DEV questions only and save scored results.

    .venv/bin/python -m extractor.run_dev 117 123

One API call (<= MAX_ATTEMPTS requests) per question id. Ids not listed in
extractor/dev_split.json are refused, so frozen test items can't be run by accident.
Results go to extractor/runs/<prompt_version>/<question_id>_<timestamp>.json.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from extractor.agent import extract
from extractor.eval import match, score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "medaesqa_v1.json"
DEV_SPLIT = Path(__file__).with_name("dev_split.json")
RUNS = Path(__file__).with_name("runs")
MAX_QUESTIONS_PER_RUN = 5  # shared rate limits: ask before running more


def run(question_id: str, question: dict) -> dict:
    result = extract(question_id, question["expert_curated_answer"])
    m = match(result.atoms, question["nuggets"])
    record = {
        "result": result.model_dump(),
        "scores": score(m, result.evidence_failures, result.hedge_flags),
        "matches": [{"atom": a.id, "nugget": n.raw, "similarity": round(s, 3)}
                    for a, n, s in m.pairs],
        "unmatched_nuggets": [n.raw for n in m.unmatched_nuggets],
        "unmatched_atoms": [a.id for a in m.unmatched_atoms],
    }
    out_dir = RUNS / result.provenance.prompt_version
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.provenance.timestamp.replace(":", "").split(".")[0]
    (out_dir / f"{question_id}_{stamp}.json").write_text(json.dumps(record, indent=2))
    return record


def main(ids: list[str]) -> None:
    dev = set(json.loads(DEV_SPLIT.read_text())["dev_question_ids"])
    not_dev = [i for i in ids if i not in dev]
    if not_dev:
        sys.exit(f"Refusing non-dev question ids {not_dev}; dev ids are {sorted(dev)}.")
    if len(ids) > MAX_QUESTIONS_PER_RUN:
        sys.exit(f"{len(ids)} questions > {MAX_QUESTIONS_PER_RUN}; ask before larger runs.")

    questions = {q["question_id"]: q for q in json.loads(DATA.read_text())}
    for qid in ids:
        try:
            rec = run(qid, questions[qid])
        except Exception as e:  # report and move on; never retry here
            print(f"{qid}: FAILED {type(e).__name__}: {str(e)[:200]}")
            continue
        print(f"{qid}: {json.dumps(rec['scores'])}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    log = logging.getLogger("faithfulmed.extractor")
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())
    log.propagate = False  # avoid duplicate lines via the root handler
    main(sys.argv[1:])
