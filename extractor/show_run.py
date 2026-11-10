"""
Print a saved Extractor run in a readable form. No API calls.

    .venv/bin/python -m extractor.show_run 117          # latest run of question 117
    .venv/bin/python -m extractor.show_run 117 --all    # every saved run of it
    .venv/bin/python -m extractor.show_run mtsamples-38 # an MTSamples run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path(__file__).with_name("runs")


def show(path: Path) -> None:
    rec = json.loads(path.read_text())
    res, scores = rec["result"], rec["scores"]
    prov = res["provenance"]
    print(f"\n=== {path.relative_to(RUNS)}")
    print(f"model {prov['model']} (served {prov['model_version']}) | "
          f"prompt {prov['prompt_version']} | temp {prov['temperature']} | seed {prov['seed']}")
    print("scores: " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in scores.items()))
    flags = set(res.get("hedge_flags", [])) | set(res["evidence_failures"])
    print("\nATOMS  (! = flagged by evidence or hedge check)")
    for a in res["atoms"]:
        mark = "!" if a["id"] in flags else " "
        hedge = a["hedge"] + (f" ({a['hedge_text']})" if a["hedge_text"] else "")
        print(f"{mark}{a['id']:>4}  [{a['type']}, {hedge}, {a['importance']}]  {a['claim']}")
    if "source" in rec:  # MTSamples: no gold nuggets
        src = rec["source"]
        print(f"\nsource: {src['dataset']} row {src['row']} ({src['specialty']}: "
              f"{src['sample_name']})")
        return
    print("\nMATCHED NUGGETS")
    for m in rec["matches"]:
        print(f"  {m['atom']:>4} <-> {m['nugget']}")
    if rec["unmatched_nuggets"]:
        print("\nUNMATCHED NUGGETS")
        for n in rec["unmatched_nuggets"]:
            print(f"   {n}")


def main(args: list[str]) -> None:
    if not args:
        sys.exit(__doc__)
    paths = sorted(RUNS.glob(f"*/{args[0]}_*.json"), key=lambda p: p.name)
    if not paths:
        sys.exit(f"No saved runs for question {args[0]} in {RUNS}")
    for p in paths if "--all" in args else paths[-1:]:
        show(p)


if __name__ == "__main__":
    main(sys.argv[1:])
