import json
import argparse
from datasets import load_dataset


# Function to transform each question into the desired format
def transform_entry(entry):
    """
    각 태스크를 llama.cpp가 요구하는 형식으로 변환합니다.

    Args:
        entry (dict): 태스크의 정보가 담긴 딕셔너리

    Returns:
        dict: llama.cpp가 요구하는 형식으로 변환된 태스크
    """
    # 정답을 인덱스로 변환
    correct_index = entry["answer"] if isinstance(entry["answer"], int) else ord(entry["answer"]) - ord('A')

    question = f"Question: {entry['question']} Answer:"
    answers = [choice + " " + str(i) for i,choice in enumerate(entry["choices"],start=1)]

    return {
        "multiple_correct": {"answers": [], "labels": []},
        "question": question,
        "single_correct": {
            "answers": answers,
            "labels": [1 if i == correct_index else 0 for i in range(len(entry["choices"]))]
        }
    }    


def parsing():

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="cais/mmlu", help="Repo. name")
    parser.add_argument("--output-path", type=str, default="mmlu.json", help="Output path")

    args = parser.parse_args()

    return args


def main(args):
    """
    데이터셋을 llama.cpp가 요구하는 형식으로 변환한 후 JSON 파일로 저장합니다.
    """

    # dataset 로드
    dataset = load_dataset(args.repo_id, "all")

    # 데이터셋을 정제
    transformed_data = [transform_entry(entry) for entry in dataset['test']]
    # 데이터를 json 파일로 저장
    with open(args.output_path, 'w') as f:
        json.dump(transformed_data, f, indent=2)

    print(f"Transformation complete. Data saved to '{args.output_path}'")


if __name__ == "__main__":
    args = parsing()
    main(args)
