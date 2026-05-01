import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict

DATA_JSON_PATH = Path(__file__).parent.parent / "data" / "HaluEval" / "data" / "summarization_data.json"

def load_haluval_data(json_path=DATA_JSON_PATH):
    """Load HaluEval summarization JSON and return a list of records.

    Each record will contain:
        - source (the knowledge string)
        - question
        - right_answer
        - hallucinated_answer
        - label (0 for faithful/right answer, 1 for hallucinated)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = []
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            source = entry.get("document", "")
            right = entry.get("right_summary", "")
            halluc = entry.get("hallucinated_summary", "")
            # Faithful example uses right_summary, hallucinated uses hallucinated_summary
            if right:
                records.append({
                    "source": source,
                    "answer": right,
                    "label": 0,
                })
            if halluc:
                records.append({
                    "source": source,
                    "answer": halluc,
                    "label": 1,
                })
    return records

def create_dataset():
    records = load_haluval_data()
    ds = Dataset.from_dict({
        "source": [r["source"] for r in records],
        "answer": [r["answer"] for r in records],
        "label": [r["label"] for r in records],
    })
    # Split 80/20
    ds = ds.train_test_split(test_size=0.2, seed=42)
    return DatasetDict({"train": ds["train"], "validation": ds["test"]})

if __name__ == "__main__":
    dataset = create_dataset()
    print(dataset)
