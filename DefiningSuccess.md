FaithfulMed — Task Specification
1. Task
Input: A de-identified or synthetic clinical document, such as a medical report, discharge instruction, or patient record.
Goal: Generate a patient-friendly explanation that:
Is understandable to a general patient.
Preserves the important facts from the original source.
Does not introduce unsupported medical claims.
Does not unnecessarily omit important information.
Avoids excessive medical jargon.
Does not refuse appropriate medical-explanation requests.
The system will eventually compare:
Single Gemini baseline
 vs.
 Multi-agent FaithfulMed pipeline
The multi-agent pipeline is:
Extractor → Simplifier → Verifier → Refiner → Readability

2. Success Rubric
The team should evaluate every generated explanation using the same metrics.
Metric
What we're measuring
Target
Flesch-Kincaid Grade Level
How difficult the explanation is to read
≤ 8th grade
SMOG
Another estimate of reading difficulty
Lower is better
Medical-jargon density
How much specialized medical terminology remains
Lower is better
Output length
Number of words in the explanation
Track, don't arbitrarily minimize
Refusal rate
% of appropriate requests where the model refuses
Lower is better
Faithfulness
Whether claims are supported by the source
≥85%
Hallucination rate
% containing a clinically meaningful unsupported claim
<10%

The project's main readability requirement is:
At least 80% of outputs should be at or below an 8th-grade Flesch-Kincaid reading level.
For faithfulness, the project targets ≥85% factual fidelity on the human-annotated test set.

3. What Each Metric Means
Flesch-Kincaid
Answers:
"What grade level would someone approximately need to understand this?"
Example:
"The medication may cause nausea."
is easier than:
"Administration of the medication may result in gastrointestinal adverse effects."
We want the generated explanation to generally be 8th grade or easier.

SMOG
SMOG is another readability measure that focuses heavily on the number of complex/multi-syllable words.
Use it as a secondary readability metric, rather than replacing Flesch-Kincaid.

Medical-jargon density
This measures how much medical terminology appears in the output.
For example:
Higher jargon:
"The patient exhibits hypertension and tachycardia."
Lower jargon:
"The patient's blood pressure and heart rate are higher than normal."
The team needs to decide which terms count as medical jargon and use that same definition for every experiment.
That's important because otherwise two people could calculate "jargon density" differently.

Output length
Simply measure the number of words.
For example:
Input: 150 words
Output: 95 words
You shouldn't automatically assume that shorter = better.
An explanation that is extremely short could have omitted an important fact.
So initially, track output length rather than making it a hard success threshold.

Refusal rate
This measures:
How often does the system refuse to explain something it should be able to explain?
For example, if you give it an ordinary lab report and it responds:
"I cannot provide medical information."
that's an unnecessary refusal.
You want to track:
Refusal rate =
unnecessary refusals / total appropriate requests

4. Shared Evaluation Harness
This is the part your team should build once and have everyone use.
Conceptually:
                   Generated Explanation
                            │
          ┌─────────────────┼─────────────────┐
          ↓                 ↓                 ↓
    Readability         Jargon             Length
          │                 │                 │
    ┌─────┴─────┐           │                 │
    ↓           ↓           ↓                 ↓
Flesch-       SMOG     Jargon Density     Word Count
Kincaid
          │
          └─────────────────┬─────────────────┘
                            ↓
                     Evaluation Results
Then later, faithfulness evaluation gets added:
Source + Explanation
        ↓
   Faithfulness
        ↓
Human labels / MedAESQA
        ↓
Accuracy + Cohen's κ

