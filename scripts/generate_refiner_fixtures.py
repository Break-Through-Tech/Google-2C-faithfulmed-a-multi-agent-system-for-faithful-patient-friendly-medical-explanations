"""Generate the frozen synthetic urology fixture set.

Run only when intentionally creating a new dataset version. Evaluation commands never mutate
fixtures. The output contains synthetic statements and no patient data.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PROVENANCE = (
    "Synthetic clinical-to-lay template annotation v2; provisional and not clinically reviewed."
)


def report(*, faithful: bool, unsupported: list[str], omissions: list[str]) -> dict:
    return {
        "faithful": faithful,
        "unsupported_claims": unsupported,
        "omissions": omissions,
        "reading_level_ok": True,
    }


def record(
    split: str,
    number: int,
    category: str,
    source: str,
    draft: str,
    verifier: dict,
    needs: bool,
    gold_flags: list[str],
    corrections: list[str],
    refined: str,
    preserve: list[str],
    unresolved: list[str] | None = None,
) -> dict:
    return {
        "example_id": f"uro_{split}_{number:03d}",
        "source_text": source,
        "draft": draft,
        "verifier_report": verifier,
        "gold_needs_refinement": needs,
        "category": category,
        "gold_flagged_issues": gold_flags,
        "expected_corrections": corrections,
        "gold_refined_text": refined,
        "preserve_facts": preserve,
        "expected_unresolved_issues": unresolved or [],
        "annotation_provenance": PROVENANCE,
    }


def build_split(split: str) -> list[dict]:
    offset = 0 if split == "dev" else 1
    rows: list[dict] = []
    number = 1

    # 10 measurements/laterality cases, including one multi-flag case.
    for i in range(10):
        if i == 4:
            source = (
                "Noncontrast CT of the abdomen and pelvis demonstrates a 4 mm calculus in the "
                "distal left ureter, without hydroureteronephrosis. The patient is afebrile."
            )
            draft = (
                "Your scan shows a 9 mm stone near the bottom of your right ureter. It is not "
                "causing kidney swelling, and you do not have a fever."
            )
            flags = ["The stone size is incorrect.", "The ureter side is incorrect."]
            refined = (
                "Your scan shows a 4 mm stone near the bottom of your left ureter. It is not "
                "causing kidney swelling, and you do not have a fever."
            )
            corrections = ["4 mm stone", "left ureter"]
        elif i < 5:
            actual, wrong = 3 + ((i + offset) % 4), 8 + ((i + offset) % 4)
            article = "an" if wrong == 8 else "a"
            source = (
                f"Noncontrast CT of the abdomen and pelvis demonstrates a {actual} mm calculus "
                "in the distal left ureter, approximately 1 cm proximal to the ureterovesical "
                "junction. No hydroureteronephrosis is identified, and the patient is afebrile."
            )
            draft = (
                f"Your scan shows {article} {wrong} mm stone near the bottom of your left ureter. It is "
                "not causing kidney swelling, and you do not have a fever."
            )
            flags = ["The stone measurement is incorrect."]
            refined = (
                f"Your scan shows a {actual} mm stone near the bottom of your left ureter. It is "
                "not causing kidney swelling, and you do not have a fever."
            )
            corrections = [f"{actual} mm stone"]
        else:
            actual = "right" if (i + offset) % 2 else "left"
            wrong = "left" if actual == "right" else "right"
            clinical, lay = [
                ("simple cortical cyst", "small fluid-filled sac"),
                ("indeterminate hypoechoic lesion", "small area that needs a closer look"),
                ("nonobstructing parenchymal calcification", "small calcium deposit"),
                ("focal cortical scarring", "small area of old scarring"),
                ("trace perinephric fluid collection", "small pocket of fluid next to the kidney"),
            ][i - 5]
            source = (
                f"Renal ultrasonography identifies a {clinical} involving the {actual} kidney. "
                "Serum creatinine remains unchanged from baseline."
            )
            draft = (
                f"The scan found a {lay} in your {wrong} kidney. Your kidney blood test has not "
                "changed."
            )
            flags = ["The kidney side is incorrect."]
            refined = (
                f"The scan found a {lay} in your {actual} kidney. Your kidney blood test has not "
                "changed."
            )
            corrections = [f"{actual} kidney"]
        rows.append(record(split, number, "measurements_laterality", source, draft,
                           report(faithful=False, unsupported=flags, omissions=[]), True, flags,
                           corrections, refined, ["you do not have a fever." if i < 5 else
                                                  "Your kidney blood test has not changed."]))
        number += 1

    # 10 negation/certainty cases.
    for i in range(10):
        if i < 5:
            clinical, lay = [
                ("gross hematuria", "blood in your urine"),
                ("dysuria", "burning when you urinate"),
                ("febrile episodes", "a fever"),
                ("costovertebral-angle pain", "pain in your side or back"),
                ("urinary retention", "trouble emptying your bladder"),
            ][i]
            source = (
                f"Review of systems is negative for {clinical}. Increased oral fluid intake was "
                "recommended as tolerated."
            )
            draft = f"You have {lay}. Try to drink more fluids if you can."
            flag = f"The draft reverses the source negation for {clinical}."
            refined = f"You do not have {lay}. Try to drink more fluids if you can."
            correction = f"do not have {lay}"
        else:
            clinical, lay = [
                ("partial ureteral obstruction", "a partly blocked urine tube"),
                ("ascending urinary tract infection", "an infection moving toward the kidney"),
                ("recent calculus passage", "a stone that recently passed"),
                ("ureteral stricture", "narrowing in the urine tube"),
                ("recurrent nephrolithiasis", "kidney stones returning"),
            ][i - 5]
            source = (
                f"The imaging appearance is nonspecific but may represent {clinical}. Interval "
                "follow-up imaging is planned."
            )
            draft = f"The scan confirms {lay}. Another scan is planned."
            flag = f"The draft overstates certainty about {clinical}."
            refined = f"The scan may show {lay}. Another scan is planned."
            correction = f"may show {lay}"
        rows.append(record(split, number, "negation_certainty", source, draft,
                           report(faithful=False, unsupported=[flag], omissions=[]), True, [flag],
                           [correction], refined, ["Try to drink more fluids if you can." if i < 5
                                                  else "Another scan is planned."]))
        number += 1

    # 10 medication/instruction cases; the last is deliberately unresolved.
    drugs = ["tamsulosin", "oxybutynin", "nitrofurantoin", "sildenafil", "finasteride"]
    for i in range(10):
        if i == 9:
            source = (
                "Continue the antimicrobial according to the existing prescription; dosing "
                "frequency is not documented in this encounter. Notify the clinic if a cutaneous "
                "eruption develops."
            )
            draft = "Take the antibiotic twice a day. Call the clinic if you develop a rash."
            flag = "The antibiotic frequency may be incorrect."
            refined = draft
            corrections: list[str] = []
            unresolved = [flag]
        elif i < 5:
            correct = drugs[(i + offset) % len(drugs)]
            wrong = drugs[(i + offset + 1) % len(drugs)]
            source = (
                f"Initiate {correct} orally once daily. Discontinue the medication and contact "
                "urology if severe orthostatic dizziness occurs."
            )
            draft = (
                f"Start {wrong} once a day. Stop taking it and call the urology clinic if you get "
                "very dizzy when standing."
            )
            flag = "The medication name is incorrect."
            refined = (
                f"Start {correct} once a day. Stop taking it and call the urology clinic if you get "
                "very dizzy when standing."
            )
            corrections = [f"Start {correct} once a day."]
            unresolved = []
        else:
            drug = drugs[(i + offset) % len(drugs)]
            source = (
                f"Administer {drug} orally every night at bedtime. Maintain the scheduled "
                "outpatient urology follow-up."
            )
            draft = f"Take {drug} twice a day. Keep your scheduled urology visit."
            flag = "The medication frequency is incorrect."
            refined = f"Take {drug} once at bedtime. Keep your scheduled urology visit."
            corrections = ["once at bedtime"]
            unresolved = []
        preserve = ["Call the clinic if you develop a rash."] if i == 9 else (
            ["call the urology clinic if you get very dizzy when standing."] if i < 5
            else ["Keep your scheduled urology visit."]
        )
        rows.append(record(split, number, "medication_instructions", source, draft,
                           report(faithful=False, unsupported=[flag], omissions=[]), True, [flag],
                           corrections, refined, preserve, unresolved))
        number += 1

    # 10 omissions/unsupported claims.
    for i in range(10):
        if i < 5:
            timing = ["one week", "two weeks", "one month", "six weeks", "three months"][i]
            source = (
                "Urinalysis is negative for leukocyte esterase, nitrites, and microscopic "
                f"hematuria. Outpatient urology reassessment is recommended in {timing}."
            )
            draft = "Your urine test did not show signs of infection or blood."
            flag = f"The urology follow-up in {timing} is omitted."
            refined = (
                f"Your urine test did not show signs of infection or blood. Follow up with "
                f"urology in {timing}."
            )
            corrections = [f"Follow up with urology in {timing}."]
            verifier = report(faithful=False, unsupported=[], omissions=[flag])
        else:
            invented = ["kidney failure", "prostate cancer", "bladder cancer", "kidney infection", "urinary blockage"][i - 5]
            source = (
                "Renal ultrasonography demonstrates no acute abnormality. Maintain usual oral "
                "hydration unless otherwise directed."
            )
            draft = (
                f"Your kidney scan did not find an urgent problem. You have {invented}. Keep "
                "drinking your usual amount of fluid unless your care team tells you otherwise."
            )
            flag = f"The claim of {invented} is unsupported."
            refined = (
                "Your kidney scan did not find an urgent problem. Keep drinking your usual "
                "amount of fluid unless your care team tells you otherwise."
            )
            corrections = ["Your kidney scan did not find an urgent problem."]
            verifier = report(faithful=False, unsupported=[flag], omissions=[])
        rows.append(record(split, number, "omissions_unsupported", source, draft, verifier, True,
                           [flag], corrections, refined,
                           ["Your urine test did not show signs of infection or blood." if i < 5
                            else "unless your care team tells you otherwise."]))
        number += 1

    # 5 false flags plus 5 missed-error negative controls. Both must be exact no-ops.
    for i in range(10):
        if i < 5:
            size = 3 + i
            source = (
                f"Noncontrast pelvic CT demonstrates a {size} mm intravesical calculus. There is "
                "no radiographic or laboratory evidence of urinary tract infection."
            )
            draft = (
                f"Your scan shows a {size} mm stone inside your bladder. There are no signs of a "
                "urinary infection."
            )
            flag = "The stone measurement is incorrect."
            verifier = report(faithful=False, unsupported=[flag], omissions=[])
        else:
            actual = 3 + (i - 5)
            wrong = 8 + (i - 5)
            article = "an" if wrong == 8 else "a"
            side = "right" if split == "test" else "left"
            source = (
                f"Noncontrast CT demonstrates a {actual} mm calculus in the distal {side} ureter "
                "without upstream hydroureteronephrosis."
            )
            draft = (
                f"Your scan shows {article} {wrong} mm stone near the bottom of your {side} ureter. It is "
                "not causing swelling in the kidney or urine tube."
            )
            flag = None
            verifier = report(faithful=True, unsupported=[], omissions=[])
        rows.append(record(split, number, "already_correct_false_flags", source, draft, verifier,
                           False, [], [], draft, [draft], []))
        number += 1

    assert len(rows) == 50
    return rows


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        path = FIXTURES / f"urology_{split}.jsonl"
        rows = build_split(split)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        print(f"wrote {len(rows)} examples to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
