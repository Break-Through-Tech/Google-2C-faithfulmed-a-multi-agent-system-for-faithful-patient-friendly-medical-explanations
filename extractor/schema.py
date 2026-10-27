"""
Extractor output schema.

Two layers:
  - AtomDraft / ExtractorOutput: what the LLM returns (passed to ADK as output_schema).
  - Atom / ExtractionResult: what code builds around it (ids, evidence check, provenance).

Ids are assigned by code, not the LLM, because LLM numbering skips and repeats.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AtomType = Literal[
    "condition",
    "symptom",
    "medication",
    "treatment",  # non-drug intervention: lifestyle, therapy, device
    "procedure",
    "test_result",
    "risk_factor",
    "instruction",
    "other",
]

# Collapsed from the ~20 hedge words used in MedAESQA nuggets.
#   possible    = may / might / can
#   frequency   = often / common / frequently / sometimes / infrequently
#   recommended = should / standard / best / recommended / try
#   negated     = not / no
Hedge = Literal["asserted", "possible", "frequency", "recommended", "negated"]

Importance = Literal["critical", "supporting"]


class AtomDraft(BaseModel):
    claim: str = Field(description="One standalone factual sentence.")
    type: AtomType
    predicate: str = Field(description="Short verb, e.g. 'treat', 'cause', 'prevent'.")
    subject: str
    object: str
    hedge: Hedge
    hedge_text: str | None = Field(
        default=None, description="Exact hedge word from the source, e.g. 'may'. Null if asserted."
    )
    evidence: str = Field(description="Verbatim quote from the source supporting the claim.")
    importance: Importance


class ExtractorOutput(BaseModel):
    atoms: list[AtomDraft]


class Atom(AtomDraft):
    id: str


class CheckedAtoms(BaseModel):
    """What downstream agents read from session state["atoms"] after the Extractor runs."""
    atoms: list[Atom]
    evidence_failures: list[str]  # atom ids whose evidence is not found in the source
    hedge_flags: list[str]  # atom ids labelled asserted but with a hedge word in evidence
    prompt_version: str
    model_version: str | None


class Provenance(BaseModel):
    model: str
    model_version: str | None
    temperature: float
    seed: int
    prompt_version: str
    timestamp: str  # UTC ISO-8601


class ExtractionResult(BaseModel):
    source_id: str
    atoms: list[Atom]
    evidence_failures: list[str]  # atom ids whose evidence is not found in the source
    hedge_flags: list[str] = []  # atom ids labelled asserted but with a hedge word in evidence
    provenance: Provenance
