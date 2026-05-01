from datasets import load_dataset, DatasetDict

def load_haluval_data(subset="summarization"):
    """Load HaluEval from Hugging Face.
    
    Subsets: summarization, qa, dialogue
    """
    print(f"Loading HaluEval {subset} dataset from Hugging Face...")
    # Using 'data' split as per HaluEval HF structure
    ds = load_dataset("pminervini/HaluEval", subset, split="data")
    
    def transform(example):
        # Summarization has right_summary, hallucinated_summary
        # QA has right_answer, hallucinated_answer
        if subset == "summarization":
            knowledge = example["document"]
            right = example["right_summary"]
            halluc = example["hallucinated_summary"]
        elif subset == "qa":
            knowledge = example["knowledge"]
            right = example["right_answer"]
            halluc = example["hallucinated_answer"]
        else: # dialogue
            knowledge = example["knowledge"]
            right = example["right_response"]
            halluc = example["hallucinated_response"]
            
        return {
            "knowledge": [knowledge, knowledge],
            "answer": [right, halluc],
            "label": [0, 1] # 0: Faithful, 1: Hallucinated
        }

    # Flatten the dataset to have one row per (knowledge, answer) pair
    records = []
    for item in ds:
        if subset == "summarization":
            records.append({"source": item["document"], "answer": item["right_summary"], "label": 0})
            records.append({"source": item["document"], "answer": item["hallucinated_summary"], "label": 1})
        elif subset == "qa":
            records.append({"source": item["knowledge"], "answer": item["right_answer"], "label": 0})
            records.append({"source": item["knowledge"], "answer": item["hallucinated_answer"], "label": 1})
        else:
            records.append({"source": item["knowledge"], "answer": item["right_response"], "label": 0})
            records.append({"source": item["knowledge"], "answer": item["hallucinated_response"], "label": 1})
            
    return records

def create_dataset(sample_size=None):
    records = load_haluval_data()
    
    import random
    if sample_size:
        random.shuffle(records)
        records = records[:sample_size]
        
    from datasets import Dataset
    ds = Dataset.from_dict({
        "source": [r["source"] for r in records],
        "answer": [r["answer"] for r in records],
        "label": [r["label"] for r in records],
    })
    
    # Split 80/20
    ds = ds.train_test_split(test_size=0.2, seed=42)
    return DatasetDict({"train": ds["train"], "validation": ds["test"]})

if __name__ == "__main__":
    dataset = create_dataset(sample_size=100)
    print(dataset)

