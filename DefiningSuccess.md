# FaithfulMed: Task Specification & Evaluation Rubric

## 🎯 Task Specification

### Objective

FaithfulMed generates **patient-friendly explanations of clinical documents** while preserving the factual information in the original source.

The system should produce explanations that are:

* **Faithful** — supported by the original clinical source
* **Readable** — targeted at an 8th-grade reading level or below
* **Patient-friendly** — understandable to a general audience
* **Concise** — avoids unnecessary information and repetition
* **Grounded** — does not introduce unsupported medical claims

### Input

The system accepts de-identified or synthetic clinical text, such as:

* Medical reports
* Discharge instructions
* Clinical notes
* Patient records
* Care plans

### Output

The system produces a plain-language explanation of the input while maintaining clinically important facts.

### System Architecture

The planned multi-agent pipeline is:

```text
Extractor → Simplifier → Verifier → Refiner → Readability
```

The multi-agent system will be compared against a **single-Gemini baseline** without specialized agents.

---

# 📊 Success Rubric

| Metric                         | Description                                                                |            Target |
| ------------------------------ | -------------------------------------------------------------------------- | ----------------: |
| **Flesch-Kincaid Grade Level** | Estimates the reading grade level of the generated explanation             |       ≤ 8th grade |
| **SMOG**                       | Secondary readability measure based largely on complex words               |   Lower is better |
| **Medical-Jargon Density**     | Percentage/proportion of words or terms identified as medical jargon       |   Lower is better |
| **Output Length**              | Number of words in the generated explanation                               | Track and compare |
| **Refusal Rate**               | Percentage of appropriate requests that receive an unnecessary refusal     |   Lower is better |
| **Faithfulness**               | Percentage of factual claims supported by the source                       |             ≥ 85% |
| **Hallucination Rate**         | Percentage of outputs containing a clinically meaningful unsupported claim |             < 10% |

### Readability Goal

At least **80% of generated outputs should have a Flesch-Kincaid reading level of 8th grade or below**.

### Faithfulness Goal

The system should achieve **at least 85% factual fidelity** on the human-annotated evaluation set.

### Verifier Goal

The Verifier should agree with human annotators at least **80% of the time**, with a target **Cohen's κ ≥ 0.6**.

### Multi-Agent Improvement Goal

The multi-agent pipeline should achieve at least a **15 percentage-point absolute improvement in faithfulness** compared with the single-agent baseline.

---

# 📏 Evaluation Metrics

## 1. Flesch-Kincaid Grade Level

Measures the approximate U.S. school grade required to understand the generated text.

**Primary readability metric.**

Lower scores indicate easier-to-read text.

**Target:** ≤ 8th grade.

---

## 2. SMOG

SMOG provides a secondary estimate of reading difficulty, with greater emphasis on complex/multi-syllable words.

Lower scores generally indicate easier-to-read text.

SMOG will be reported alongside Flesch-Kincaid rather than used as the primary readability criterion.

---

## 3. Medical-Jargon Density

Measures the amount of specialized medical terminology remaining in the generated explanation.

Example:

> "The patient exhibits hypertension and tachycardia."

versus:

> "The patient's blood pressure and heart rate are higher than normal."

The team will establish a consistent definition and calculation method for medical jargon before evaluating the frozen test set.

**Important:** The same jargon definition must be applied across all models and pipeline variants.

---

## 4. Output Length

Measures the number of words in the generated explanation.

Output length will initially be treated as a **tracked metric rather than a hard optimization target**.

Shorter output is not necessarily better if important clinical information is omitted.

---

## 5. Refusal Rate

Measures how frequently the system refuses requests that it should be capable of answering.

For example, an unnecessary refusal to explain a standard medical report would count as a refusal.

```text
Refusal Rate =
Unnecessary Refusals / Appropriate Requests
```

Lower refusal rates are preferred.

---

# 🔬 Faithfulness Evaluation

Faithfulness measures whether information in the generated explanation is supported by the original source.

The primary evaluation dataset is **MedAESQA**, which contains:

* 40 health questions
* Expert answers
* Machine-generated answers
* Evidence excerpts
* Human accuracy/evidence-support judgments
* Expert "nuggets" representing atomic facts

MedAESQA will be used for:

* Verifier calibration
* Extractor evaluation
* Faithfulness evaluation

**MedAESQA is an evaluation dataset, not a training dataset.**

---

# 🧪 Shared Evaluation Harness

All models and pipeline configurations should be evaluated using the same evaluation harness.

The harness should calculate:

```text
Input
  │
  ▼
Generated Explanation
  │
  ├── Flesch-Kincaid Grade Level
  ├── SMOG
  ├── Medical-Jargon Density
  ├── Output Length
  ├── Refusal Rate
  │
  └── Faithfulness Evaluation
          │
          ├── Accuracy
          ├── Cohen's κ
          ├── Omission
          └── Unsupported Claims / Hallucination
```

## Recommended Evaluation Record

Each evaluated example should store results in a consistent format:

```text
example_id
model
pipeline
flesch_kincaid_grade
smog_score
jargon_density
word_count
refusal
faithfulness_score
hallucination
```

This allows results from different models, agents, and ablations to be compared directly.

---

# 🏆 Example Results Table

The following numbers are **illustrative only** and should not be used as project results.

| Model/Pipeline               | FK Grade | SMOG | Jargon Density | Words | Refusal Rate | Faithfulness |
| ---------------------------- | -------: | ---: | -------------: | ----: | -----------: | -----------: |
| Single Gemini Baseline       |        — |    — |              — |     — |            — |            — |
| FaithfulMed                  |        — |    — |              — |     — |            — |            — |
| FaithfulMed - No Extractor   |        — |    — |              — |     — |            — |            — |
| FaithfulMed - No Simplifier  |        — |    — |              — |     — |            — |            — |
| FaithfulMed - No Verifier    |        — |    — |              — |     — |            — |            — |
| FaithfulMed - No Refiner     |        — |    — |              — |     — |            — |            — |
| FaithfulMed - No Readability |        — |    — |              — |     — |            — |            — |

---

# 🔒 Evaluation Protocol

To ensure fair comparisons:

1. Use the **same frozen evaluation set** for every model and pipeline.
2. Use the **same metric definitions** for every experiment.
3. Use the same evaluation code whenever possible.
4. Record every experiment and result.
5. Do not modify the gold set after evaluation begins.
6. Report headline numbers on the frozen gold-standard set.
7. Clearly distinguish between development examples and test examples.

---

# 📈 Research Questions

The evaluation framework is designed to answer:

1. Does the multi-agent pipeline produce more readable explanations than a single Gemini baseline?
2. Does the multi-agent pipeline produce more faithful explanations?
3. Which agents contribute most to **faithfulness**?
4. Which agents contribute most to **readability**?
5. Does the Verifier successfully identify unsupported claims?
6. What tradeoffs exist between readability, faithfulness, latency, cost, and model openness?

---

# 🔬 Ablation Analysis

Each agent will be removed individually to measure its marginal contribution.

```text
Full Pipeline
      │
      ├── Remove Extractor
      ├── Remove Simplifier
      ├── Remove Verifier
      ├── Remove Refiner
      └── Remove Readability
```

The resulting performance will be compared against the full pipeline.

This analysis will help determine **which agents actually matter** for faithfulness and readability.

---

# 📌 Key Principle

> **Build the evaluation harness before optimizing the agents.**

The evaluation framework must remain consistent throughout the project so that improvements can be measured reliably and reproducibly.


