"""
Run the LOCKED Extractor on the MedAESQA test questions (all ids not in dev_split.json).

    .venv/bin/python -m extractor.run_test --write-lock   # freeze code -> test_lock.json (commit it)
    .venv/bin/python -m extractor.run_test                # run every test question not yet done
    .venv/bin/python -m extractor.run_test --summary      # aggregate saved results, no API calls

Refuses to run if the prompt, schema, checks or scoring code differ from test_lock.json, so
test results always come from the code that was committed before any test item was seen.
Resumable: questions already saved are skipped, so a 503 only costs a re-run of that item.
One API call (<= MAX_ATTEMPTS requests) per question, with a pause between calls.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

from extractor.agent import DEFAULT_MODEL, DEFAULT_SEED, TEMPERATURE, extract
from extractor.eval import MATCHER_VERSION, match, score
from extractor.prompt import PROMPT_VERSION

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "medaesqa_v1.json"
LOCK = HERE / "test_lock.json"
LOCKED_FILES = ["prompt.py", "schema.py", "checks.py", "preprocess.py", "eval.py", "agent.py"]
PAUSE_S = 5.0  # be gentle with the team's shared per-minute quota


def code_hash() -> str:
    h = hashlib.sha256()
    for name in LOCKED_FILES:
        h.update(name.encode() + b"\0" + (HERE / name).read_bytes())
    return h.hexdigest()[:16]


def current_config() -> dict:
    return {"code_hash": code_hash(), "prompt_version": PROMPT_VERSION,
            "matcher_version": MATCHER_VERSION, "model": DEFAULT_MODEL,
            "temperature": TEMPERATURE, "seed": DEFAULT_SEED}


def test_ids() -> list[str]:
    dev = set(json.loads((HERE / "dev_split.json").read_text())["dev_question_ids"])
    return sorted((q["question_id"] for q in json.loads(DATA.read_text())
                   if q["question_id"] not in dev), key=int)


def out_dir(lock: dict) -> Path:
    return HERE / "runs" / "test" / f"{lock['prompt_version']}_{lock['code_hash']}"


def check_lock() -> dict:
    if not LOCK.exists():
        sys.exit("No test_lock.json. Run --write-lock and commit it before any test run.")
    lock = json.loads(LOCK.read_text())
    now = current_config()
    diff = {k: (lock.get(k), v) for k, v in now.items() if lock.get(k) != v}
    if diff:
        sys.exit(f"Code or config changed since the lock, refusing: {diff}")
    return lock


def run_all() -> None:
    lock = check_lock()
    out = out_dir(lock)
    out.mkdir(parents=True, exist_ok=True)
    questions = {q["question_id"]: q for q in json.loads(DATA.read_text())}
    todo = [i for i in test_ids() if not (out / f"{i}.json").exists()]
    print(f"{len(todo)} test questions to run -> {out}")
    for n, qid in enumerate(todo, 1):
        q = questions[qid]
        try:
            result = extract(qid, q["expert_curated_answer"])
        except Exception as e:  # leave it undone; a later run picks it up
            print(f"[{n}/{len(todo)}] {qid}: FAILED {type(e).__name__}: {str(e)[:120]}")
            time.sleep(PAUSE_S)
            continue
        m = match(result.atoms, q["nuggets"])
        rec = {
            "lock": lock,
            "result": result.model_dump(),
            "scores": score(m, result.evidence_failures, result.hedge_flags),
            "matches": [{"atom": a.id, "nugget": nu.raw, "similarity": round(s, 3)}
                        for a, nu, s in m.pairs],
            "unmatched_nuggets": [nu.raw for nu in m.unmatched_nuggets],
            "unmatched_atoms": [a.id for a in m.unmatched_atoms],
        }
        (out / f"{qid}.json").write_text(json.dumps(rec, indent=2))
        s = rec["scores"]
        print(f"[{n}/{len(todo)}] {qid}: recall {s['nugget_recall']:.2f} "
              f"precision>={s['atom_precision_lower_bound']:.2f} "
              f"evidence_fail {s['evidence_fail_rate']:.2f}")
        time.sleep(PAUSE_S)
    summary()


def summary() -> None:
    lock = json.loads(LOCK.read_text())
    recs = [json.loads(p.read_text()) for p in sorted(out_dir(lock).glob("*.json"))]
    expected = len(test_ids())
    if not recs:
        sys.exit("No saved test results yet.")
    tot = {k: sum(r["scores"][k] for r in recs) for k in ("n_matched", "n_nuggets", "n_atoms")}
    ev = sum(len(r["result"]["evidence_failures"]) for r in recs)
    hf = sum(len(r["result"]["hedge_flags"]) for r in recs)
    hedge = [r["scores"]["hedge_accuracy"] for r in recs if r["scores"]["hedge_accuracy"] is not None]
    recall = [r["scores"]["nugget_recall"] for r in recs]
    print(f"lock {lock['code_hash']} | {lock['model']} | {lock['prompt_version']} | "
          f"{lock['matcher_version']}")
    print(f"questions scored: {len(recs)}/{expected}")
    print(f"nugget recall (micro): {tot['n_matched']}/{tot['n_nuggets']} = "
          f"{tot['n_matched'] / tot['n_nuggets']:.3f}")
    print(f"nugget recall (macro): {sum(recall) / len(recall):.3f} "
          f"(min {min(recall):.2f}, max {max(recall):.2f})")
    print(f"atom precision lower bound (micro): {tot['n_matched']}/{tot['n_atoms']} = "
          f"{tot['n_matched'] / tot['n_atoms']:.3f}")
    print(f"evidence failures: {ev}/{tot['n_atoms']} atoms = {ev / tot['n_atoms']:.3f}")
    print(f"hedge flags: {hf}/{tot['n_atoms']} atoms = {hf / tot['n_atoms']:.3f}")
    print(f"hedge agreement with annotators (macro): {sum(hedge) / len(hedge):.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    if sys.argv[1:] == ["--write-lock"]:
        LOCK.write_text(json.dumps({**current_config(),
                                    "test_question_ids": test_ids()}, indent=2) + "\n")
        print(f"wrote {LOCK}: {json.loads(LOCK.read_text())['code_hash']}")
    elif sys.argv[1:] == ["--summary"]:
        summary()
    else:
        run_all()
