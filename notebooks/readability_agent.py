""" 
FaithfulMed - Readability Agent

Evaluates and imrpoves the readability of medical explanations
while attempting to preserve their clincial meaning.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext

from eval_harness import readability_scores, meets_readability_target

#Configuration
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
TARGET_GRADE = 8.0
MAX_REWRITES = 2

# Instructions shared by Gemini and the ADK agent
READABILITY_INSTRUCTIONS = """
You are the Readability Agent for FaithfulMed.

Your responsbility is to make medical explanations easier for
patients to understand.

Guidelines:
- Target an 8th-grade reading level or below.
- Use short, clear sentences.
- Replace complex medical jargon with simpler language.
- Explain necessary medical terminology.
- Maintain a a clear, professional, and approachable tone.
- Preserve all important clinical information.
- Do not change diagnoses, medical names, dosages, numbers,
warnings, or other treatment instructions.
- Do not invent facts or introduce new medical advice.
- Do not omit clinically important information.

Writing perspective:
- Preserve the original grammatical point of view.
- Do not change "the patient" into "you."
- Do not turn confirmed statements into hypothetical statements.
- Maintain the original level of medical certainty.

Medical terminology:
- Prefer everyday words over complex medical terminology.
- Replace medical jargon with plain-language explanations
  when the original term is not necessary.
- Do not repeat difficult terms in parentheses merely
  to define them.
- Retain medical terms only when necessary for accuracy,
  identification, or important clinical instructions.
- If a medical term must be retained, explain it briefly
  using simple language.
- Use short sentences instead of long definitions.
- Do not add new clinical facts or unsupported details.


Return only the rewritten explanation, without any additional commentary or formatting.
"""


def record_readability(callback_context: CallbackContext):
    """Evaluate the Readability Agent's final output."""

    final_text = callback_context.state.get("final")

    if not isinstance(final_text, str) or not final_text.strip():
        print("No final text available for readability scoring.")
        return None

    scores = evaluate_readability(final_text)

    callback_context.state["readability_scores"] = scores
    callback_context.state["readability_passed"] = scores["meets_target"]

    print("\nADK READABILITY EVALUATION:")
    print(scores)

    return None

#ADK agent for integration with the team pipeline
readability_agent = LlmAgent(
    name="Readability",
    model=MODEL,
    instruction=(
        READABILITY_INSTRUCTIONS
        + "\nRewrite the following corrected draft:\n{draft}\n"
        + "\nUse these extracted clinical facts as reference:\n{atoms}"
    ),
    output_key="final",
    after_agent_callback=record_readability,
)

def evaluate_readability(text: str) -> dict:
    """Calculate readability scores and whether the target is met."""
    
    scores = readability_scores(text)
    scores["meets_target"] = meets_readability_target(
        text, max_grade=TARGET_GRADE
    )
    
    return scores

def simplify_text(text: str, client, model: str = MODEL, current_grade: float | None = None) -> str:
    """Rewrite medical text using readability feedback."""

    feedback = ""

    if current_grade is not None:
        feedback = (
            f"\nCurrent Flesch-Kincaid grade: {current_grade:.2f}. "
            f"Target grade: {TARGET_GRADE:.1f} or below.\n"
            "Use shorter sentences and simpler words. "
            "Avoid unnecessary medical jargon and long explanations. "
            "Preserve the original POV and all clinical facts.\n"
        )

    response = client.models.generate_content(
        model=model,
        contents=(
            READABILITY_INSTRUCTIONS
            + feedback
            + "\nRewrite this medical explanation:\n\n"
            + text
        ),
    )

    if not response.text or not response.text.strip():
        raise ValueError("Gemini returned an empty response.")

    return response.text.strip()

def process_readability(text: str, client, model: str = MODEL) -> dict:
    """
    Improve readability with a maximum of two rewrites.

    Keeps track of the lowest reading-grade candidate.
    Clinical faithfulness must be checked separately.
    """

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    original_scores = evaluate_readability(text)

    # If already readable, don't change anything.
    if original_scores["meets_target"]:
        return {
            "original_text": text,
            "final_text": text,
            "original_scores": original_scores,
            "final_scores": original_scores,
            "rewrite_attempts": 0,
            "meets_target": True,
            "requires_faithfulness_check": False,
        }

    current_text = text
    current_scores = original_scores

    best_text = text
    best_scores = original_scores

    attempts = 0

    while attempts < MAX_REWRITES:
        rewritten_text = simplify_text(
            current_text,
            client,
            model,
            current_grade=current_scores["flesch_kincaid_grade"]
        )

        attempts += 1
        rewritten_scores = evaluate_readability(rewritten_text)

        # Retain the most readable candidate found so far.
        if (
            rewritten_scores["flesch_kincaid_grade"]
            < best_scores["flesch_kincaid_grade"]
        ):
            best_text = rewritten_text
            best_scores = rewritten_scores

        # Stop as soon as a candidate meets the target.
        if rewritten_scores["meets_target"]:
            break

        # Use the best available version for the next attempt.
        current_text = best_text
        current_scores = best_scores

    return {
        "original_text": text,
        "final_text": best_text,
        "original_scores": original_scores,
        "final_scores": best_scores,
        "rewrite_attempts": attempts,
        "meets_target": best_scores["meets_target"],
        "requires_faithfulness_check": best_text != text,
    }