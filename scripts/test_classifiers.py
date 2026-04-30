import json
import os
import numpy as np
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# 1. CONFIGURATION
MODELS_TO_TEST = [
    "best_model_bert-base-cased",
    "best_model_microsoft-deberta-v3-base",
    "best_model_answerdotai-ModernBERT-base"
]

tokenizer_names = {
    "best_model_bert-base-cased": "bert-base-cased",
    "best_model_microsoft-deberta-v3-base": "microsoft/deberta-v3-base",
    "best_model_answerdotai-ModernBERT-base": "answerdotai/ModernBERT-base"
}


MAX_LENGTH = 256
BATCH_SIZE = 32
RESULTS_FILE = "xeval_benchmarks.json"

# TEST_FILE = "eurollmAWQbacktranslated/backtranslated_results_EuroLLM-9B-Instruct-AWQ_emotional_pl_to_en.json"
# TEST_FILE = "ayaexpanseAWQbacktranslated/backtranslated_results_aya-expanse-8b-AWQ_emotional_de_to_en.json"
TEST_FILE = "gemmaAWQbacktranslated/backtranslated_results_gemma-2-9b-it-AWQ_emotional_de_to_en.json"

# 2. DATA PREPROCESSING (reuse logic from training)
def load_and_prepare_data():
    with open("train_dataset.json", "r") as f:
        train_raw = json.load(f)
    with open(TEST_FILE, "r") as f:
        test_raw = json.load(f)

    mlb = MultiLabelBinarizer()
    all_labels = [item["labels"] for item in train_raw]
    mlb.fit(all_labels)

    def transform_set(data):
        return {
            "text": [item["text"] for item in data],
            "label": mlb.transform([item["labels"] for item in data]).astype(float).tolist(),
        }

    return (
        Dataset.from_dict(transform_set(train_raw)),
        Dataset.from_dict(transform_set(test_raw)),
        mlb,
    )


train_ds, test_ds, mlb = load_and_prepare_data()
num_labels = len(mlb.classes_)
print(f"Number of labels: {num_labels}")


# 3. METRIC CALCULATION (same as in training)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs > 0.5).astype(float)
    return {
        "f1_micro": f1_score(labels, predictions, average="micro"),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "accuracy": accuracy_score(labels, predictions),
    }


# 4. EVALUATION LOOP (load fine-tuned models and evaluate on test set)
all_results = {}

for model_name in MODELS_TO_TEST:
    print(f"\n{'=' * 20}\nEVALUATING (fine-tuned): {model_name}\n{'=' * 20}")

    #clean_name = model_name.replace("/", "-")
    #final_model_path = f"./best_model_{clean_name}"
    #eval_output_dir = f"./eval_{clean_name}"

    if not os.path.isdir(model_name):
        print(f"Skipped {model_name}: fine-tuned model not found at {model_name}")
        continue

    # Use the same base tokenizer as during training
    #print(model_name)
    #exit()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_names[model_name], use_fast=False)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized_test = test_ds.map(tokenize_fn, batched=True)

    # Load fine-tuned model from disk
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir="./out_dir",
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    # Global metrics
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


# 5. SAVE EVAL SCORES
with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"\nAll fine-tuned models evaluated. Results saved to {RESULTS_FILE}")

