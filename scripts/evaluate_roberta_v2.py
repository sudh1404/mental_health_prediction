
import json
from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TEST_CSV = (
    ROOT / "data" / "raw" / "dreaddit" / "dreaddit-test.csv"
)

MODEL_DIR = ROOT / "models" / "roberta_stress_v2"

METRICS_DIR = ROOT / "results" / "metrics"
CONFUSION_DIR = ROOT / "results" / "confusion_matrices"
PREDICTIONS_DIR = ROOT / "results" / "predictions"

OFFICIAL_METRICS = (
    METRICS_DIR / "roberta_stress_metrics.json"
)

CONFUSION_PATH = (
    CONFUSION_DIR / "roberta_v2_confusion_matrix.png"
)

PREDICTIONS_PATH = (
    PREDICTIONS_DIR / "roberta_v2_test_predictions.csv"
)

MAX_LENGTH = 256
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATA CLEANING
# Keep consistent with the V2 training pipeline
# ============================================================

def clean_data(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {path}"
        )

    df = pd.read_csv(path)

    df = df[["text", "label"]].dropna()

    df["text"] = df["text"].astype(str).str.strip()

    df = df[df["text"] != ""]

    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected class labels in test dataset.")

    return df.reset_index(drop=True)


# ============================================================
# DATASET
# ============================================================

class StressTestDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }

        item["labels"] = self.labels[index]

        return item


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, loader):

    model.eval()

    predictions = []
    actual_labels = []
    probabilities_list = []

    with torch.inference_mode():

        for batch in loader:

            labels = batch.pop("labels")

            inputs = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

            predicted = torch.argmax(
                probabilities,
                dim=-1
            )

            predictions.extend(
                predicted.cpu().tolist()
            )

            actual_labels.extend(
                labels.tolist()
            )

            probabilities_list.extend(
                probabilities.cpu().tolist()
            )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            actual_labels,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    metrics = {
        "model": "roberta-base",
        "model_version": "V2",
        "selected_seed": 42,
        "task": "binary_stress_detection",
        "accuracy": float(
            accuracy_score(actual_labels, predictions)
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "test_examples": len(actual_labels),
        "max_length": MAX_LENGTH,
        "checkpoint": "models/roberta_stress_v2",
        "evaluation_device": str(DEVICE)
    }

    return (
        metrics,
        actual_labels,
        predictions,
        probabilities_list
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def generate_confusion_matrix(actual, predictions):

    cm = confusion_matrix(
        actual,
        predictions,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Stress", "Stress"]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True
    )

    ax.set_title(
        "RoBERTa V2 (Seed 42) - Stress Detection"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    plt.tight_layout()

    plt.savefig(
        CONFUSION_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Confusion matrix saved: {CONFUSION_PATH}")

    return cm


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

def save_predictions(
    df,
    actual,
    predictions,
    probabilities
):

    prediction_df = pd.DataFrame({

        "text": df["text"].tolist(),

        "actual_label": actual,

        "predicted_label": predictions,

        "actual_class": [
            "Stress" if x == 1 else "No Stress"
            for x in actual
        ],

        "predicted_class": [
            "Stress" if x == 1 else "No Stress"
            for x in predictions
        ],

        "no_stress_probability": [
            round(float(p[0]), 6)
            for p in probabilities
        ],

        "stress_probability": [
            round(float(p[1]), 6)
            for p in probabilities
        ]

    })

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    print(f"Predictions saved: {PREDICTIONS_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n======================================")
    print("       ROBERTA V2 EVALUATION")
    print("======================================")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"V2 checkpoint not found: {MODEL_DIR}"
        )

    # Load test dataset
    print("\nLoading Dreaddit test dataset...")

    test_df = clean_data(TEST_CSV)

    print(f"Test examples: {len(test_df)}")

    # Load tokenizer and model
    print("\nLoading RoBERTa V2...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    if model.config.num_labels != 2:
        raise ValueError("Expected a binary classification model.")

    if (
        model.config.id2label.get(0) != "No Stress"
        or model.config.id2label.get(1) != "Stress"
    ):
        raise ValueError(
            "Unexpected model label mapping. "
            "Check config.json before evaluation."
        )

    model.to(DEVICE)
    model.eval()

    print("RoBERTa V2 loaded successfully!")
    print(f"Device: {DEVICE}")

    # Prepare dataset
    dataset = StressTestDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Evaluate
    print("\nEvaluating V2 on test dataset...")

    (
        metrics,
        actual,
        predictions,
        probabilities
    ) = evaluate_model(model, loader)

    # Display metrics
    print("\n========== V2 TEST RESULTS ==========")

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]:

        print(
            f"{key}: {metrics[key]:.4f}"
        )

    print(
        f"Test examples: {metrics['test_examples']}"
    )

    # Generate confusion matrix
    print("\nGenerating confusion matrix...")

    cm = generate_confusion_matrix(
        actual,
        predictions
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Save predictions
    save_predictions(
        test_df,
        actual,
        predictions,
        probabilities
    )

    # Save official metrics
    with open(
        OFFICIAL_METRICS,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"\nOfficial metrics saved: {OFFICIAL_METRICS}"
    )

    print("\n======================================")
    print("       EVALUATION COMPLETED")
    print("======================================")


if __name__ == "__main__":
    main()
