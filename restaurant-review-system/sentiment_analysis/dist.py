#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
    pipeline
)
import evaluate


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./yelp_distilbert_output"
SEED = 42
EPOCHS = 2
BATCH_TRAIN = 32
BATCH_EVAL = 64
LR = 5e-5
MAX_LEN = 128

FAST_DEBUG = False
FAST_TRAIN_N = 80_000
FAST_VALID_N = 8_000

USE_FP16 = torch.cuda.is_available()

set_seed(SEED)


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("Loading Yelp Polarity...")
ds = load_dataset("yelp_polarity")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=MAX_LEN)

ds = ds.map(tokenize, batched=True)
ds = ds.rename_column("label", "labels")

split = ds["train"].train_test_split(test_size=0.02, seed=SEED)
train_ds, valid_ds = split["train"], split["test"]
test_ds = ds["test"]

if FAST_DEBUG:
    train_ds = train_ds.select(range(min(FAST_TRAIN_N, len(train_ds))))
    valid_ds = valid_ds.select(range(min(FAST_VALID_N, len(valid_ds))))
    print(f"[FAST] Using {len(train_ds)} train / {len(valid_ds)} valid")

cols = ["input_ids", "attention_mask", "labels"]
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in cols])
valid_ds = valid_ds.remove_columns([c for c in valid_ds.column_names if c not in cols])
test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in cols])

collator = DataCollatorWithPadding(tok)


# ============================================================================
# METRICS
# ============================================================================

metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": metric_f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, "| FP16:", USE_FP16)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_TRAIN,
    per_device_eval_batch_size=BATCH_EVAL,
    gradient_accumulation_steps=1,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=USE_FP16,
    logging_steps=200,
    report_to="none",
    save_total_limit=2
)

early_stop = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.0005
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=collator,
    tokenizer=tok,
    compute_metrics=compute_metrics,
    callbacks=[early_stop]
)


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

trainer.train()

print("\nValidation metrics:")
val_metrics = trainer.evaluate(eval_dataset=valid_ds)
print(val_metrics)

print("\nTest metrics:")
test_metrics = trainer.evaluate(eval_dataset=test_ds)
print(test_metrics)


# ============================================================================
# MODEL SAVING
# ============================================================================

save_path = os.path.join(OUTPUT_DIR, "best_model")
trainer.save_model(save_path)
tok.save_pretrained(save_path)
print(f"✅ Saved best model to: {save_path}")


# ============================================================================
# INFERENCE DEMO
# ============================================================================

clf = pipeline(
    "text-classification",
    model=save_path,
    tokenizer=save_path,
    device=0 if torch.cuda.is_available() else -1
)

samples = [
    "This restaurant was amazing, loved the service and the food!",
    "Terrible experience. The staff was rude and the food was cold."
]

print("\nSample predictions:")
for s in samples:
    print(s, "->", clf(s))
