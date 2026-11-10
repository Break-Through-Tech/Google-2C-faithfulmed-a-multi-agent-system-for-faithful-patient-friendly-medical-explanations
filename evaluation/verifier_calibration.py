import json

# Load MedAESQA
with open("data/medaesqa_v1.json", "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} questions.")

# Show the human accuracy labels
for question in data[:5]:
    for model_id, answer in question["machine_generated_answers"].items():
        print(
            question["question_id"],
            model_id,
            answer["is_answer_accurate"]
        )