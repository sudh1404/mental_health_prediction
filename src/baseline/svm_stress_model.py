
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
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

for directory in [MODEL_DIR, METRIC_DIR, FIGURE_DIR, PREDICTION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("MODEL 2: TF-IDF + LINEAR SVM")
    print("=" * 60)

    # 1. Load the SAME dataset used by Logistic Regression
    print("\n[1] Loading Dreaddit...")
    train_df, test_df = load_dreaddit()

    # 2. Apply the SAME preprocessing
    print("[2] Preprocessing...")
    train_df = preprocess_dataframe(train_df)
    test_df = preprocess_dataframe(test_df)

    X_train = train_df["cleaned_text"]
    y_train = train_df["label"]

    X_test = test_df["cleaned_text"]
    y_test = test_df["label"]

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # 3. Create the SVM pipeline
    print("\n[3] Creating model...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=False,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=42,
                max_iter=5000,
            ),
        ),
    ])

    # 4. Train
    print("[4] Training Linear SVM...")
    model.fit(X_train, y_train)
    print("Training complete!")

    # 5. Predict
    print("\n[5] Evaluating...")
    predictions = model.predict(X_test)

    # SVM decision scores are NOT probabilities.
    decision_scores = model.decision_function(X_test)

    # 6. Calculate metrics
    metrics = {
        "model": "TF-IDF + Linear SVM",
        "task": "Binary Stress Detection",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
    }

    print("\n" + "=" * 60)
    print("SVM RESULTS")
    print("=" * 60)

    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:12s}: {value:.4f}")

    print("\nClassification Report:\n")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No Stress", "Stress"],
            zero_division=0,
        )
    )

    # 7. Confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Stress", "Stress"],
        yticklabels=["No Stress", "Stress"],
    )
    plt.title("TF-IDF + Linear SVM: Stress Detection")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    figure_path = FIGURE_DIR / "svm_stress_confusion_matrix.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    # 8. Save metrics
    metrics_path = METRIC_DIR / "svm_stress_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # 9. Save predictions
    prediction_df = pd.DataFrame({
        "text": test_df["text"].values,
        "actual_label": y_test.values,
        "predicted_label": predictions,
        "decision_score": decision_scores,
    })

    predictions_path = (
        PREDICTION_DIR / "svm_stress_predictions.csv"
    )
    prediction_df.to_csv(predictions_path, index=False)

    # 10. Save trained model
    model_path = MODEL_DIR / "tfidf_linear_svm_stress.joblib"
    joblib.dump(model, model_path)

    print("\n" + "=" * 60)
    print("SVM BASELINE COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {figure_path}")
    print(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
