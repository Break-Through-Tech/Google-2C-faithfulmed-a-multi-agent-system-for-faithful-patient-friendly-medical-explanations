
import sys
from pathlib import Path

from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

# Find the notebooks folder so Python can import the existing agent
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from readability_agent import process_readability
from google import genai

sample_text = """
Diabetes mellitus is a chronic metabolic disorder characterized
by elevated blood glucose levels due to impaired insulin secretion
or insulin resistance.
"""

def main():
    client = genai.Client()

    try:
        result = process_readability(sample_text, client)

        print("\n--- Simplified Text ---")
        print(result["final_text"])

        print("\n--- Original Scores ---")
        print(result["original_scores"])

        print("\n--- Final Scores ---")
        print(result["final_scores"])

        print("\n--- Meets Target? ---")
        print(result["meets_target"])

        print("\n--- Rewrite Attempts ---")
        print(result["rewrite_attempts"])

    finally:
        client.close()

if __name__ == "__main__":
    main()
