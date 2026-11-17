from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.data_loader import load_dreaddit
from src.preprocessing import preprocess_dataframe


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "baseline_stress"
METRIC_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
PREDICTION_DIR = PROJECT_ROOT / "results" / "predictions"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("BASELINE STRESS DETECTION")
    print("TF-IDF + LOGISTIC REGRESSION")
    print("=" * 70)

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    print("\n[1] Loading dataset...")

    train_df, test_df = load_dreaddit()

    print("Training samples:", len(train_df))
    print("Test samples    :", len(test_df))

    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    print("\n[2] Preprocessing...")

    train_df = preprocess_dataframe(train_df)
    test_df = preprocess_dataframe(test_df)

    X_train = train_df["cleaned_text"]
    y_train = train_df["label"]

    X_test = test_df["cleaned_text"]
    y_test = test_df["label"]

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    print("\n[3] Creating TF-IDF + Logistic Regression model...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=False,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    print("\n[4] Training...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    print("\n[5] Generating predictions...")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Stress",
                "Stress"
            ],
            zero_division=0
        )
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "No Stress",
            "Stress"
        ],
        yticklabels=[
            "No Stress",
            "Stress"
        ]
    )

    plt.title(
        "TF-IDF + Logistic Regression\nStress Detection"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    confusion_path = (
        FIGURE_DIR /
        "baseline_stress_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300
    )

    plt.close()

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "task": "Stress Detection",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    metrics_path = (
        METRIC_DIR /
        "baseline_stress_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # -----------------------------------------------------
    # SAVE PREDICTIONS
    # -----------------------------------------------------

    prediction_df = test_df[
        ["text", "label"]
    ].copy()

    prediction_df["predicted_label"] = predictions

    prediction_df["no_stress_probability"] = (
        probabilities[:, 0]
    )

    prediction_df["stress_probability"] = (
        probabilities[:, 1]
    )

    prediction_path = (
        PREDICTION_DIR /
        "baseline_stress_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    model_path = (
        MODEL_DIR /
        "tfidf_logistic_stress.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    # -----------------------------------------------------
    # FINISHED
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE COMPLETE")
    print("=" * 70)

    print("\nSaved files:")

    print("\nModel:")
    print(model_path)

    print("\nMetrics:")
    print(metrics_path)

    print("\nConfusion matrix:")
    print(confusion_path)

    print("\nPredictions:")
    print(prediction_path)


if __name__ == "__main__":
    main()