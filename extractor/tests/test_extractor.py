"""Offline tests for the Extractor. No API calls."""
import json
from pathlib import Path

import pytest

from extractor.checks import assign_ids, check_evidence, check_hedges
from extractor.eval import map_hedge, match, parse_nugget, score
from extractor.preprocess import preprocess
from extractor.schema import AtomDraft, ExtractorOutput

DATA = Path(__file__).resolve().parents[2] / "data" / "medaesqa_v1.json"


def draft(**overrides) -> AtomDraft:
    base = dict(type="treatment", predicate="treat", subject="CPAP", object="sleep apnea",
                hedge="asserted", hedge_text=None,
                evidence="CPAP is the standard treatment", importance="critical")
    base.update(overrides)
    base.setdefault("claim", f"{base['subject']} {base['predicate']} {base['object']}.")
    return AtomDraft(**base)


# --- preprocess ---------------------------------------------------------------

def test_preprocess_strips_pmid_brackets():
    text = "Lifestyle changes may help [28646811, 33659106]. CPAP works [28901030]."
    assert preprocess(text) == "Lifestyle changes may help. CPAP works."


def test_preprocess_keeps_non_pmid_brackets():
    assert preprocess("dose [see label] daily") == "dose [see label] daily"


# --- checks -------------------------------------------------------------------

def test_assign_ids_sequential():
    atoms = assign_ids([draft(), draft()])
    assert [a.id for a in atoms] == ["A1", "A2"]


def test_check_evidence_flags_paraphrase_only():
    source = "While CPAP is the standard treatment [28901030], surgery was also suggested."
    atoms = assign_ids([
        draft(),                                       # verbatim -> ok
        draft(evidence="cpap  is the STANDARD treatment."),  # case/space/period -> ok
        draft(evidence="CPAP is the best treatment"),  # paraphrase -> fail
        draft(evidence=""),                            # empty -> fail
    ])
    assert check_evidence(atoms, preprocess(source)) == ["A3", "A4"]


def test_check_evidence_ellipsis_fragments_must_be_verbatim_and_in_order():
    source = "Mutations can cause bone deformities, height lower than expected, extra teeth."
    atoms = assign_ids([
        draft(evidence="Mutations can cause... extra teeth"),           # ok: both in order
        draft(evidence="Mutations can cause\u2026 height lower"),       # ok: unicode ellipsis
        draft(evidence="extra teeth... Mutations can cause"),          # out of order -> fail
        draft(evidence="Mutations can cause... purple teeth"),         # invented -> fail
        draft(evidence="..."),                                         # nothing -> fail
    ])
    assert check_evidence(atoms, preprocess(source)) == ["A3", "A4", "A5"]


def test_check_hedges_flags_dropped_may():
    atoms = assign_ids([
        draft(hedge="asserted", evidence="part of the organ may come through the cut"),
        draft(hedge="possible", hedge_text="may", evidence="surgery may remove the cyst"),
        draft(hedge="asserted", evidence="CPAP is the standard treatment"),
        draft(hedge="asserted", evidence="a canal in the ear"),  # "can" inside a word
        draft(hedge="asserted", evidence="This gene mostly affects bone development"),
    ])
    assert check_hedges(atoms) == ["A1", "A5"]


def test_schema_rejects_unknown_type():
    with pytest.raises(ValueError):
        draft(type="diagnosis")


def test_output_schema_parses_json():
    raw = json.dumps({"atoms": [draft().model_dump()]})
    assert len(ExtractorOutput.model_validate_json(raw).atoms) == 1


# --- eval ---------------------------------------------------------------------

def test_parse_nugget_with_hedge():
    n = parse_nugget("[may] Treat (drugs [C1547857], sleep apnea [C0037315])")
    assert n.parsed and n.hedge == "possible" and n.predicate == "treat"
    assert n.args == ["drugs", "sleep apnea"]


def test_parse_nugget_parenthesised_hedge_falls_back_to_full_text():
    # "(primary)" uses parentheses, not brackets, so it is matched on its full text.
    n = parse_nugget("(primary) Treat (lifestyle [C0870811] and dietary changes "
                     "[C3671772], all IBS [C0022104] patients)")
    assert n.parsed is False and "C0870811" not in n.args[0]


def test_parse_nugget_if_then_is_full_text():
    n = parse_nugget("IF Treat (laparoscopy [C0031150], ovarian cyst [C0029927]) THEN [may] "
                     "Form (small number of patients, ovarian cyst again after laparoscopy)")
    assert n.parsed is False


def test_parse_nugget_unparseable_is_kept():
    n = parse_nugget("IF injury [C0521711] is greater than 4mm THEN see a doctor")
    assert n.parsed is False and "[C0521711]" not in n.args[0]


def test_map_hedge():
    assert map_hedge(None) == "asserted"
    assert map_hedge("most common") == "frequency"
    assert map_hedge("reported") is None


def test_match_and_score():
    atoms = assign_ids([
        draft(subject="CPAP", object="obstructive sleep apnea", hedge="recommended"),
        draft(subject="drugs", object="sleep apnea", hedge="asserted"),  # hedge wrong
        draft(subject="massage", object="snoring"),                       # not a nugget
    ])
    nuggets = [
        "[standard] Treat (CPAP [C0199451], obstructive sleep apnea [C0520679])",
        "[may] Treat (drugs [C1547857], sleep apnea [C0037315])",
        "Prevent (quitting smoking [C0085134], sleep apnea [C0037315])",
    ]
    s = score(match(atoms, nuggets), evidence_failures=["A3"])
    assert s["n_matched"] == 2
    assert s["nugget_recall"] == pytest.approx(2 / 3)
    assert s["atom_precision_lower_bound"] == pytest.approx(2 / 3)
    assert s["hedge_accuracy"] == pytest.approx(1 / 2)
    assert s["evidence_fail_rate"] == pytest.approx(1 / 3)


def test_match_action_in_argument():
    # q133 failure under matcher v1: the action is in the nugget's subject slot.
    atoms = assign_ids([draft(subject="surgery", predicate="remove", object="the cyst",
                              claim="Surgery may remove the cyst.")])
    m = match(atoms, ["[may] Treat (Remove [C0195488], ovarian cyst [C0029927])"])
    assert len(m.pairs) == 1


def test_match_if_then_on_claim_text():
    atoms = assign_ids([draft(subject="cysts", predicate="form", object="again",
                              claim="Cysts may form again after laparoscopy in a small "
                                    "number of patients.")])
    m = match(atoms, ["IF Treat (laparoscopy [C0031150], ovarian cyst [C0029927]) THEN [may] "
                      "Form (small number of patients, ovarian cyst again after laparoscopy)"])
    assert len(m.pairs) == 1


def test_sibling_atom_does_not_steal_missing_nugget():
    # The extractor found "extra teeth" but missed "bone deformities". The sibling shares
    # subject and predicate, but must not be credited with the missing nugget.
    atoms = assign_ids([draft(subject="RUNX2 mutations", predicate="cause",
                              object="extra teeth")])
    nuggets = ["Cause (runx2 [C1419771] mutation [C0596611], extra teeth [C0040457])",
               "Cause (runx2 [C1419771] mutation [C0596611], bone deformities [C0410719])"]
    m = match(atoms, nuggets)
    assert [n.raw for _, n, _ in m.pairs] == [nuggets[0]]
    assert [n.raw for n in m.unmatched_nuggets] == [nuggets[1]]


@pytest.mark.skipif(not DATA.exists(), reason="medaesqa_v1.json not present")
def test_every_gold_nugget_parses_without_error():
    """Read-only pass over the frozen gold set: nothing is dropped or modified."""
    data = json.loads(DATA.read_text())
    nuggets = [s for q in data for s in q["nuggets"]]
    parsed = [parse_nugget(s) for s in nuggets]
    assert len(parsed) == len(nuggets) == 498
    assert sum(not n.parsed for n in parsed) == 17  # IF/THEN + comparisons: full-text match


# --- after_agent_callback (pipeline path) --------------------------------------

def _fake_ctx(state: dict, source: str):
    from types import SimpleNamespace

    from google.genai import types

    return SimpleNamespace(
        state=state,
        user_content=types.Content(role="user", parts=[types.Part(text=source)]),
        agent_name="Extractor",
    )


def test_postprocess_atoms_writes_checked_atoms_to_state():
    from extractor.agent import postprocess_atoms
    from extractor.schema import CheckedAtoms

    source = "Surgery may remove the cyst [123]. CPAP is the standard treatment."
    raw = {"atoms": [
        draft(evidence="Surgery may remove the cyst", hedge="asserted").model_dump(),
        draft(evidence="CPAP is the best treatment").model_dump(),
    ]}
    state = {"atoms": raw, "extractor_model_version": "test-model-001"}
    assert postprocess_atoms(_fake_ctx(state, source)) is None

    checked = CheckedAtoms.model_validate(state["atoms"])
    assert [a.id for a in checked.atoms] == ["A1", "A2"]
    assert checked.evidence_failures == ["A2"]  # paraphrased quote
    assert checked.hedge_flags == ["A1"]        # "may" in evidence, labelled asserted
    assert checked.model_version == "test-model-001"
    assert checked.prompt_version.startswith("extractor-v")


def test_postprocess_atoms_accepts_json_string_and_fails_on_bad_output():
    from extractor.agent import postprocess_atoms

    state = {"atoms": json.dumps({"atoms": [draft().model_dump()]})}
    postprocess_atoms(_fake_ctx(state, "CPAP is the standard treatment"))
    assert state["atoms"]["atoms"][0]["id"] == "A1"

    with pytest.raises(ValueError):  # truncated JSON must not pass downstream
        postprocess_atoms(_fake_ctx({"atoms": '{"atoms": [{"claim": "CP'}, "x"))
    with pytest.raises(RuntimeError):
        postprocess_atoms(_fake_ctx({}, "x"))


def test_agent_registers_callback():
    from extractor.agent import build_extractor_agent, postprocess_atoms

    agent = build_extractor_agent()
    assert agent.after_agent_callback is postprocess_atoms
    assert agent.output_key == "atoms"
