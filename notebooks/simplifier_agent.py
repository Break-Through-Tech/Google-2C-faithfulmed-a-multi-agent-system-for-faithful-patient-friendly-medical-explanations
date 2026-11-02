"""
FaithfulMed - Simplifier Agent

Rewrites the Extractor's clinical atoms in plain language, grounded by
lay definitions from the MedlinePlus retrieval index.

Flow:
    atoms -> find complicated terms in each atom (Gemini)
          -> retrieve a MedlinePlus definition for each term (Qdrant)
          -> give Gemini each atom + its definitions -> plain-language atom
          -> aggregate the plain atoms into one patient-facing draft

Gemini calls per document: 2 (term finding and rewriting are batched over
all atoms). Embedding calls: one per unique term.
"""
import os
import re
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.adk.models.llm_response import LlmResponse
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext

#Configuration
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# The retrieval package lives at the repo root, not in notebooks/.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.retriever import RetrievalResult, retrieve  # noqa: E402

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

DEFINITIONS_PER_TERM = 2   # top-k chunks retrieved per term
MIN_SIMILARITY = 0.55      # drop weak matches; not yet calibrated
MAX_DEFINITION_CHARS = 600 # health-topic chunks can be long

TERM_INSTRUCTIONS = """
You are helping simplify medical text for patients.

For each numbered clinical fact below, list the medical terms a patient
with an 8th-grade reading level might not understand: diagnoses, drug
classes, procedures, anatomy, lab names, and abbreviations.

Rules:
- Copy each term exactly as it appears in the fact.
- Keep multi-word terms together (e.g. "lower-extremity edema").
- Do not list everyday words, numbers, doses, or units.
- A fact with no difficult terms gets an empty list.

Return JSON: [{"index": <fact number>, "terms": ["...", ...]}, ...]
"""

REWRITE_INSTRUCTIONS = """
You are the Simplifier Agent for FaithfulMed.

Rewrite each numbered clinical fact so a patient at an 8th-grade reading
level can understand it.

Grounding:
- Each fact comes with MedlinePlus definitions for its difficult terms.
- Use those definitions only to explain what a term means.
- Do not copy extra facts from the definitions (causes, treatments,
  statistics) into the rewrite.
- If a term has no definition, explain it briefly in plain words or keep
  it as is. Never guess at its meaning.

Faithfulness:
- Each rewrite must say the same thing as its fact: nothing added,
  nothing left out.
- Keep drug names, doses, numbers, units, dates, and lab values exactly.
- Keep the original level of certainty and point of view.
- Do not change "the patient" into "you."
- Do not give new medical advice.

Style:
- Short sentences and everyday words.
- One to three sentences per fact.

Return JSON: [{"index": <fact number>, "plain_text": "..."}, ...]
with exactly one entry for every fact.
"""


def parse_atoms(atoms) -> list[dict]:
    """
    Normalize the Extractor's output into a list of {text, type} dicts.

    Accepts a list, or the JSON string an LlmAgent writes to state
    (optionally wrapped in ```json fences).
    """

    if isinstance(atoms, str):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", atoms.strip())
        atoms = json.loads(cleaned)

    if not isinstance(atoms, list) or not atoms:
        raise ValueError("Simplifier Agent requires a nonempty list of atoms.")

    parsed = []

    for atom in atoms:
        if isinstance(atom, str):
            atom = {"text": atom, "type": ""}

        text = str(atom.get("text", "")).strip()

        if text:
            parsed.append({"text": text, "type": atom.get("type", "")})

    if not parsed:
        raise ValueError("Every atom was empty.")

    return parsed


def _numbered(atoms: list[dict]) -> str:
    return "\n".join(f"{i}. {atom['text']}" for i, atom in enumerate(atoms))


def _generate_json(client, model: str, prompt: str) -> list[dict]:
    """Call Gemini in JSON mode and return the parsed list."""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    if not response.text or not response.text.strip():
        raise ValueError("Gemini returned an empty response.")

    result = json.loads(response.text)

    if not isinstance(result, list):
        raise ValueError(f"Expected a JSON list from Gemini, got: {result!r}")

    return result


def extract_terms(atoms: list[dict], client, model: str = MODEL) -> list[list[str]]:
    """Find the complicated terms in every atom with one Gemini call."""

    result = _generate_json(
        client,
        model,
        TERM_INSTRUCTIONS + "\nFacts:\n" + _numbered(atoms),
    )

    terms: list[list[str]] = [[] for _ in atoms]

    for entry in result:
        index = entry.get("index")

        if isinstance(index, int) and 0 <= index < len(atoms):
            terms[index] = [
                str(term).strip()
                for term in entry.get("terms", [])
                if str(term).strip()
            ]

    return terms


def lookup_definitions(terms: list[str], retriever=None) -> dict[str, list[RetrievalResult]]:
    """
    Retrieve MedlinePlus definitions for each unique term.

    Terms are deduplicated case-insensitively so a repeated term costs one
    embedding call. Matches below MIN_SIMILARITY are dropped.
    """

    # Resolved at call time so tests can patch the module-level retrieve.
    retriever = retriever or retrieve
    definitions: dict[str, list[RetrievalResult]] = {}

    for term in terms:
        key = term.lower()

        if key in definitions:
            continue

        hits = retriever(term, k=DEFINITIONS_PER_TERM)
        definitions[key] = [hit for hit in hits if hit.similarity >= MIN_SIMILARITY]

    return definitions


def _format_definitions(terms: list[str], definitions: dict[str, list[RetrievalResult]]) -> str:
    if not terms:
        return "   (no difficult terms)"

    lines = []

    for term in terms:
        hits = definitions.get(term.lower(), [])

        if not hits:
            lines.append(f'   - "{term}": no definition found')
            continue

        for hit in hits:
            text = hit.text[:MAX_DEFINITION_CHARS]
            lines.append(f'   - "{term}" -> MedlinePlus "{hit.title}": {text}')

    return "\n".join(lines)


def rewrite_atoms(
    atoms: list[dict],
    atom_terms: list[list[str]],
    definitions: dict[str, list[RetrievalResult]],
    client,
    model: str = MODEL,
) -> list[str]:
    """Rewrite every atom in plain language with one Gemini call."""

    blocks = [
        f"{i}. {atom['text']}\n"
        f"   Definitions:\n{_format_definitions(atom_terms[i], definitions)}"
        for i, atom in enumerate(atoms)
    ]

    result = _generate_json(
        client,
        model,
        REWRITE_INSTRUCTIONS + "\nFacts:\n\n" + "\n\n".join(blocks),
    )

    rewrites: dict[int, str] = {}

    for entry in result:
        index = entry.get("index")
        text = str(entry.get("plain_text", "")).strip()

        if isinstance(index, int) and 0 <= index < len(atoms) and text:
            rewrites[index] = text

    missing = [i for i in range(len(atoms)) if i not in rewrites]

    # A dropped atom is an omission, which the Verifier would flag anyway.
    if missing:
        raise ValueError(f"Gemini did not rewrite atoms {missing}.")

    return [rewrites[i] for i in range(len(atoms))]


def aggregate(plain_atoms: list[str]) -> str:
    """Join the plain-language atoms, in source order, into one draft."""

    return " ".join(plain_atoms)


def process_simplification(atoms, client, model: str = MODEL, retriever=None) -> dict:
    """
    Run the full Simplifier: terms -> definitions -> rewrite -> aggregate.

    Returns the draft plus per-atom provenance (which MedlinePlus pages
    grounded which term) for the Verifier and for citations.
    """

    atoms = parse_atoms(atoms)
    atom_terms = extract_terms(atoms, client, model)

    all_terms = [term for terms in atom_terms for term in terms]
    definitions = lookup_definitions(all_terms, retriever)

    plain_atoms = rewrite_atoms(atoms, atom_terms, definitions, client, model)

    simplified_atoms = [
        {
            "source_text": atom["text"],
            "type": atom["type"],
            "plain_text": plain,
            "definitions": {
                term: [
                    {"title": hit.title, "url": hit.url, "similarity": hit.similarity}
                    for hit in definitions.get(term.lower(), [])
                ]
                for term in terms
            },
        }
        for atom, terms, plain in zip(atoms, atom_terms, plain_atoms)
    ]

    return {
        "draft": aggregate(plain_atoms),
        "simplified_atoms": simplified_atoms,
    }


async def run_simplifier_adk(callback_context: CallbackContext, llm_request) -> LlmResponse:
    """Run the simplifier logic inside ADK instead of a free LLM call."""

    atoms = callback_context.state.get("atoms")

    # Retrieval and Gemini calls are blocking; keep them off ADK's event loop.
    def process():
        client = genai.Client()
        try:
            return process_simplification(atoms, client)
        finally:
            client.close()

    result = await asyncio.to_thread(process)

    # Provenance for the Verifier and the demo's citations.
    callback_context.state["simplified_atoms"] = result["simplified_atoms"]

    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text=result["draft"])]
        )
    )

#ADK agent for integration with the team pipeline

simplifier_agent = LlmAgent(name="Simplifier", model=MODEL, instruction=(
        "Rewrite the extracted clinical atoms in plain language for a "
        "patient, using MedlinePlus definitions and adding nothing new.\n"
        "Atoms:\n{atoms}"
    ),
    output_key="draft",
    before_model_callback=run_simplifier_adk,
)
