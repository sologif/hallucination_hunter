import os
import sys
import argparse
from pathlib import Path

# Add the project root to sys.path so we can import from scripts
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from scripts.prepare_haluval import create_dataset
import torch
import numpy as np

def train(sample_size=100):
    # 1. Load dataset
    print(f"Preparing dataset with sample size {sample_size}...")
    dataset_dict = create_dataset(sample_size=sample_size)
    
    # model_name = "cross-encoder/nli-deberta-v3-base" # Strongest base
    # We use 'small' for the actual training in this environment to ensure it doesn't crash
    model_name = "cross-encoder/nli-deberta-v3-small" 
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Preprocess
    def preprocess_function(examples):
        return tokenizer(
            examples["source"],
            examples["answer"],
            truncation=True,
            padding="max_length",
            max_length=256 # Reduced from 512 for speed
        )

    print("Tokenizing dataset...")
    tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

    # 3. Load model for binary classification
    print("Loading model for binary classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )

    # 4. Training Arguments - Optimized for fast demo
    training_args = TrainingArguments(
        output_dir="./models/nli_finetuned",
        eval_strategy="no",
        save_strategy="no",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=10,
        report_to="none",
        no_cuda=True if not torch.cuda.is_available() else False # Use CPU if no GPU
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        processing_class=tokenizer,
    )

    # 6. Train and Save
    print("Starting training...")
    trainer.train()
    
    print("Saving fine-tuned model...")
    os.makedirs("./models/nli_finetuned", exist_ok=True)
    trainer.save_model("./models/nli_finetuned")
    tokenizer.save_pretrained("./models/nli_finetuned")
    
    # Write metadata for the UI
    with open("./models/nli_finetuned/metadata.json", "w") as f:
        json.dump({
            "trained_on": "HaluEval (Hugging Face)",
            "sample_size": sample_size,
            "base_model": model_name
        }, f)
        
    print(f"Training complete! Model saved to ./models/nli_finetuned")

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description="Fine-tune NLI model on HaluEval")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of examples to use for training")
    args = parser.parse_args()
    
    train(sample_size=args.sample_size)

