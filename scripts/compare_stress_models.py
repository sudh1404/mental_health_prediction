
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

V2_METRICS_FILE = (
    PROJECT_ROOT
    / "models"
    / "roberta_stress_v2"
    / "roberta_stress_metrics.json"
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL METRICS CONFIGURATION
# ============================================================

model_files = {
    "Logistic Regression": METRICS_DIR / "baseline_stress_metrics.json",
    "Linear SVM": METRICS_DIR / "svm_stress_metrics.json",
    "RoBERTa V2 (Seed 42)": V2_METRICS_FILE,
}


# ============================================================
# LOAD METRICS
# ============================================================

def load_metrics(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

rows = []

for model_name, path in model_files.items():

    metrics = load_metrics(path)

    f1 = metrics.get(
        "f1",
        metrics.get("f1_score")
    )

    if f1 is None:
        raise KeyError(
            f"F1 score missing in {path.name}"
        )

    rows.append({
        "Model": model_name,
        "Accuracy": float(metrics["accuracy"]),
        "Precision": float(metrics["precision"]),
        "Recall": float(metrics["recall"]),
        "F1-score": float(f1),
    })


comparison_df = pd.DataFrame(rows)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========== STRESS MODEL COMPARISON ==========\n")

print(
    comparison_df.to_string(
        index=False,
        float_format="%.4f"
    )
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = METRICS_DIR / "stress_model_comparison.csv"

comparison_df.to_csv(
    csv_path,
    index=False
)

print(f"\nComparison CSV saved: {csv_path}")


# ============================================================
# VISUALIZE PERFORMANCE
# ============================================================

ax = comparison_df.set_index("Model").plot(
    kind="bar",
    figsize=(11, 6),
    rot=0,
    width=0.75
)

ax.set_title(
    "Stress Detection: Model Performance Comparison"
)

ax.set_xlabel("Model")
ax.set_ylabel("Score")

ax.set_ylim(0, 1)

ax.legend(title="Evaluation Metric")

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

figure_path = (
    FIGURES_DIR / "stress_model_comparison.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Comparison figure saved: {figure_path}")

print("\n========== COMPARISON COMPLETED ==========")
