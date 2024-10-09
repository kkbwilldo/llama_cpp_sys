from datasets import load_dataset

# Load the MMLU dataset
dataset = load_dataset("cais/mmlu", "all")

# Function to transform each question into the desired format
def transform_entry(entry):
    # Check if the answer is an integer and use it directly as the correct index
    correct_index = entry["answer"] if isinstance(entry["answer"], int) else ord(entry["answer"]) - ord('A')
    
    return {
        "multiple_correct": {"answers": [], "labels": []},
        "question": entry["question"],
        "single_correct": {
            "answers": entry["choices"],
            "labels": [1 if i == correct_index else 0 for i in range(len(entry["choices"]))]
        }
    }

# Apply transformation to all entries
transformed_data = [transform_entry(entry) for entry in dataset['test']]

# Save transformed data to a JSON file
import json
with open('mmlu.json', 'w') as f:
    json.dump(transformed_data, f, indent=2)

print("Transformation complete. Data saved to 'mmlu.json'")
