
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "raw" / "dreaddit" / "dreaddit-train.csv"
TEST_CSV = ROOT / "data" / "raw" / "dreaddit" / "dreaddit-test.csv"

# Save to a separate folder first. This does not overwrite the existing model.
OUTPUT_DIR = ROOT / "models" / "roberta_stress_new"

BASE_MODEL = "FacebookAI/roberta-base"
MAX_LENGTH = 96
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 2e-5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")


class StressDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item


def clean_data(path):
    df = pd.read_csv(path)
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {path}")

    return df


def evaluate(model, loader):
    model.eval()
    predictions = []
    actual = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)

            predictions.extend(outputs.logits.argmax(dim=1).cpu().tolist())
            actual.extend(labels.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predictions, average="binary", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(actual, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def main():
    start = time.time()

    print("Loading Dreaddit data...")
    train_df = clean_data(TRAIN_CSV)
    test_df = clean_data(TEST_CSV)

    # Reserve part of the training data for validation.
    fit_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        random_state=SEED,
        stratify=train_df["label"],
    )

    print(f"Training examples: {len(fit_df)}")
    print(f"Validation examples: {len(val_df)}")
    print(f"Test examples: {len(test_df)}")
    print("Device: CPU")
    print("Downloading/loading pretrained RoBERTa. Internet is required on first run.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "No Stress", 1: "Stress"},
        label2id={"No Stress": 0, "Stress": 1},
    )
    model.to(device)

    train_ds = StressDataset(
        fit_df["text"].tolist(),
        fit_df["label"].tolist(),
        tokenizer,
    )
    val_ds = StressDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
    )
    test_ds = StressDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch, labels=labels)
            outputs.loss.backward()
            optimizer.step()

            running_loss += outputs.loss.item()

            if step % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS}, "
                    f"step {step}/{len(train_loader)}, "
                    f"loss {running_loss / 50:.4f}"
                )
                running_loss = 0.0

        print("Validation:", evaluate(model, val_loader))

    print("Evaluating on the separate Dreaddit test file...")
    test_metrics = evaluate(model, test_loader)
    print("Test metrics:", json.dumps(test_metrics, indent=2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(
        OUTPUT_DIR / "roberta_stress_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "model": BASE_MODEL,
                "task": "binary_stress_detection",
                "training_examples": len(fit_df),
                "validation_examples": len(val_df),
                "test_examples": len(test_df),
                **test_metrics,
            },
            file,
            indent=2,
        )

    print(f"\nSaved trained model to: {OUTPUT_DIR}")
    print(f"Elapsed time: {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
