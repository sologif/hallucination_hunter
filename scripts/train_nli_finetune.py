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

def train(sample_size=None):
    # 1. Load dataset
    print("Preparing dataset...")
    dataset_dict = create_dataset()
    
    if sample_size:
        print(f"Sampling {sample_size} examples for training...")
        dataset_dict["train"] = dataset_dict["train"].shuffle(seed=42).select(range(min(sample_size, len(dataset_dict["train"]))))
        dataset_dict["validation"] = dataset_dict["validation"].shuffle(seed=42).select(range(min(sample_size // 5, len(dataset_dict["validation"]))))
    
    model_name = "cross-encoder/nli-deberta-v3-base"
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Preprocess
    def preprocess_function(examples):
        # We pair source/knowledge with the answer
        return tokenizer(
            examples["source"],
            examples["answer"],
            truncation=True,
            padding="max_length",
            max_length=512
        )

    print("Tokenizing dataset...")
    tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

    # 3. Load model for binary classification
    # Label 0: Faithful, Label 1: Hallucinated
    print("Loading model for binary classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="./models/nli_finetuned",
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1, # Start with 1 epoch for full dataset on CPU
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. Train and Save
    print("Starting training...")
    trainer.train()
    
    print("Saving fine-tuned model...")
    trainer.save_model("./models/nli_finetuned")
    tokenizer.save_pretrained("./models/nli_finetuned")
    print("Training complete! Model saved to ./models/nli_finetuned")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune NLI model on HaluEval")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of examples to use for training")
    args = parser.parse_args()
    
    train(sample_size=args.sample_size)
