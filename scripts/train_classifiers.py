import json
import os
import shutil
import numpy as np
import torch
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# 1. CONFIGURATION
MODELS_TO_TEST = [
    "microsoft/deberta-v3-base",
    "bert-base-cased",
    "answerdotai/ModernBERT-base"
    # "bert-base-multilingual-cased",
    # "microsoft/mdeberta-v3-base",
    # "xlm-roberta-base"
]
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 10
RESULTS_FILE = "model_benchmarks.json"

# 2. DATA PREPROCESSING (Same as before)
def load_and_prepare_data():
    with open("train_dataset.json", 'r') as f: train_raw = json.load(f)
    with open("test_dataset.json", 'r') as f: test_raw = json.load(f)
    
    mlb = MultiLabelBinarizer()
    all_labels = [item['labels'] for item in train_raw]
    mlb.fit(all_labels)
    
    def transform_set(data):
        return {
            "text": [item['text'] for item in data],
            "label": mlb.transform([item['labels'] for item in data]).astype(float).tolist()
        }

    return Dataset.from_dict(transform_set(train_raw)), \
           Dataset.from_dict(transform_set(test_raw)), \
           mlb

train_ds, test_ds, mlb = load_and_prepare_data()
num_labels = len(mlb.classes_)
print(f"Number of labels: {num_labels}")

# 3. METRIC CALCULATION
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs > 0.5).astype(float)
    return {
        "f1_micro": f1_score(labels, predictions, average='micro'),
        "f1_macro": f1_score(labels, predictions, average='macro'),
        "accuracy": accuracy_score(labels, predictions)
    }

# 4. TRAINING LOOP
all_results = {}

for model_name in MODELS_TO_TEST:
    print(f"\n{'='*20}\nSTARTING: {model_name}\n{'='*20}")
    
    clean_name = model_name.replace("/", "-")
    output_dir = f"./checkpoints_{clean_name}"
    final_model_path = f"./best_model_{clean_name}"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    tokenized_train = train_ds.map(tokenize_fn, batched=True)
    tokenized_test = test_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        save_total_limit=1, # Only keep the best checkpoint on disk
        report_to="none"    # Prevents unnecessary logging clutter
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    # Run Training
    train_output = trainer.train()
    
    # Evaluate and capture scores (global metrics)
    eval_results = trainer.evaluate()

    # Per-class metrics on the test set
    preds_output = trainer.predict(tokenized_test)
    logits = preds_output.predictions
    labels = preds_output.label_ids
    probs = 1 / (1 + np.exp(-logits))
    y_pred = (probs > 0.5).astype(int)
    y_true = labels

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class = {}
    for idx, label_name in enumerate(mlb.classes_):
        per_class[str(label_name)] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    eval_results["per_class"] = per_class
    all_results[model_name] = eval_results
    
    # Save the absolute best version and cleanup temp checkpoints
    trainer.save_model(final_model_path)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    print(f"Finished {model_name}. Best F1-Micro: {eval_results['eval_f1_micro']:.4f}")

# 5. SAVE FINAL SCORES
with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"\nAll models trained. Comparison results saved to {RESULTS_FILE}")
