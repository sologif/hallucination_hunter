import json
import argparse
import sys
from tqdm import tqdm
from engine import analyze_hallucination

def load_data(file_path, sample_size=None):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}")
        print("Please ensure the HaluEval repository is cloned into the data/ directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    return data

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hallucination Hunter against HaluEval")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of examples to evaluate (default: 50)")
    parser.add_argument("--dataset", type=str, default="data/HaluEval/data/qa_data.json", help="Path to HaluEval JSON dataset or HF identifier (google-research/true)")
    parser.add_argument("--subset", type=str, default="qags_cnndm", help="Subset for TRUE dataset (default: qags_cnndm)")
    args = parser.parse_args()

    dataset = []
    if args.dataset == "google-research/true":
        try:
            from datasets import load_dataset
            print(f"Loading TRUE dataset ({args.subset}) from Hugging Face...")
            ds = load_dataset(args.dataset, args.subset, split="train", streaming=True)
            for i, item in enumerate(ds):
                if i >= args.sample_size:
                    break
                dataset.append(item)
        except Exception as e:
            print(f"Error loading TRUE dataset: {e}")
            return
    else:
        print(f"Loading {args.sample_size} examples from {args.dataset}...")
        dataset = load_data(args.dataset, args.sample_size)
    
    if not dataset:
        print("No data loaded. Exiting.")
        return

    y_true = []
    y_pred = []
    
    # We map FAITHFUL -> 0 (Not hallucinated), HALLUCINATED -> 1
    label_map = {"FAITHFUL": 0, "HALLUCINATED": 1}

    print("Starting evaluation...")
    for item in tqdm(dataset, desc="Evaluating"):
        # Handle TRUE format
        if "premise" in item and "hypothesis" in item:
            knowledge = item["premise"]
            claim = item["hypothesis"]
            label = item.get("label", 1) # In TRUE, 1 is Entailment, 0 is Not Entailment
            
            res = analyze_hallucination(knowledge, claim)
            # Map labels to binary: 0=Faithful, 1=Hallucinated
            # TRUE Label 1 (Faithful) -> 0
            # TRUE Label 0 (Hallucinated) -> 1
            y_true.append(1 if label == 0 else 0)
            y_pred.append(1 if res["verdict"] == "HALLUCINATED" else 0)
            continue

        # Handle HaluEval formats
        knowledge = item.get("knowledge", item.get("document", ""))
        right_answer = item.get("right_answer", item.get("right_summary", ""))
        hallucinated_answer = item.get("hallucinated_answer", item.get("hallucinated_summary", ""))
        
        if not knowledge:
            continue
            
        # 1. Test Faithful Example (Ground Truth = 0)
        if right_answer:
            res_faithful = analyze_hallucination(knowledge, right_answer)
            y_true.append(0)
            y_pred.append(label_map.get(res_faithful["verdict"], 1))
            
        # 2. Test Hallucinated Example (Ground Truth = 1)
        if hallucinated_answer:
            res_hallucinated = analyze_hallucination(knowledge, hallucinated_answer)
            y_true.append(1)
            y_pred.append(label_map.get(res_hallucinated["verdict"], 1))

    # Calculate metrics
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # also known as True Positive Rate or Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # True Negative Rate
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    balanced_acc = (recall + specificity) / 2

    print("\n" + "="*60)
    print(" HALLUCINATION HUNTER - EVALUATION RESULTS ")
    print("="*60)
    print(f"Total Inferences Run: {len(y_true)}")
    print(f"Dataset Used:         {args.dataset}")
    print("-"*60)
    print(f"True Positives (Correctly caught hallucination):  {tp}")
    print(f"True Negatives (Correctly verified faithful):     {tn}")
    print(f"False Positives (Falsely flagged faithful):       {fp}")
    print(f"False Negatives (Missed hallucination):           {fn}")
    print("-"*60)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f} (How many flagged were actually hallucinations)")
    print(f"Recall:            {recall:.4f} (How many of total hallucinations were caught)")
    print(f"F1 Score:          {f1:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"\n>> Balanced Accuracy: {balanced_acc:.4f} <<")
    print("="*60)

if __name__ == "__main__":
    main()
