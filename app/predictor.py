
from pathlib import Path

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_DIR = PROJECT_ROOT / "models" / "baseline_stress"

# Updated to the selected RoBERTa V2 checkpoint
ROBERTA_DIR = PROJECT_ROOT / "models" / "roberta_stress_v2"


# ============================================================
# STRESS PREDICTOR
# ============================================================

class StressPredictor:

    def __init__(self):

        print("\n========== INITIALIZING STRESS PREDICTOR ==========")

        # ----------------------------------------------------
        # Load Logistic Regression
        # ----------------------------------------------------

        print("\nLoading Logistic Regression...")

        self.lr_model = joblib.load(
            BASELINE_DIR / "tfidf_logistic_stress.joblib"
        )

        # ----------------------------------------------------
        # Load Linear SVM
        # ----------------------------------------------------

        print("Loading Linear SVM...")

        self.svm_model = joblib.load(
            BASELINE_DIR / "tfidf_linear_svm_stress.joblib"
        )

        # ----------------------------------------------------
        # Device Configuration
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        self.tokenizer = None
        self.roberta_model = None

        # ----------------------------------------------------
        # Load RoBERTa V2
        # ----------------------------------------------------

        print("\nLoading RoBERTa V2...")
        print(f"Model directory: {ROBERTA_DIR}")

        if not ROBERTA_DIR.exists():
            raise FileNotFoundError(
                f"RoBERTa V2 directory not found: {ROBERTA_DIR}"
            )

        try:

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(ROBERTA_DIR),
                local_files_only=True
            )

            self.roberta_model = (
                AutoModelForSequenceClassification.from_pretrained(
                    str(ROBERTA_DIR),
                    local_files_only=True
                )
            )

            self.roberta_model.to(self.device)
            self.roberta_model.eval()

            print("RoBERTa V2 loaded successfully!")

            print(
                "Model classes:",
                self.roberta_model.config.num_labels
            )

            print(
                "Model label mapping:",
                self.roberta_model.config.id2label
            )

        except (OSError, ValueError) as exc:

            self.tokenizer = None
            self.roberta_model = None

            print(f"RoBERTa V2 unavailable: {exc}")
            print(
                "Continuing with Logistic Regression "
                "and Linear SVM."
            )

        print("\nAll available models initialized successfully!")

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, text):

        if not text or not text.strip():
            raise ValueError(
                "Please enter a social media post."
            )

        text = text.strip()

        # ----------------------------------------------------
        # Baseline Predictions
        # ----------------------------------------------------

        lr_prediction = int(
            self.lr_model.predict([text])[0]
        )

        svm_prediction = int(
            self.svm_model.predict([text])[0]
        )

        labels = {
            0: "No Stress",
            1: "Stress"
        }

        # ----------------------------------------------------
        # Initial Result
        # ----------------------------------------------------

        result = {
            "text": text,

            "logistic_regression": {
                "prediction": labels[lr_prediction]
            },

            "linear_svm": {
                "prediction": labels[svm_prediction]
            },

            "roberta": {
                "prediction": "Unavailable",
                "confidence": None,
                "no_stress_probability": None,
                "stress_probability": None
            }
        }

        # ----------------------------------------------------
        # RoBERTa V2 Prediction
        # ----------------------------------------------------

        if (
            self.tokenizer is not None
            and self.roberta_model is not None
        ):

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():

                outputs = self.roberta_model(**inputs)

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=-1
                )[0]

            prediction = int(
                torch.argmax(probabilities).item()
            )

            result["roberta"] = {
                "prediction": labels[prediction],

                "confidence": round(
                    float(probabilities[prediction].item()),
                    4
                ),

                "no_stress_probability": round(
                    float(probabilities[0].item()),
                    4
                ),

                "stress_probability": round(
                    float(probabilities[1].item()),
                    4
                )
            }

        return result


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    predictor = StressPredictor()

    sample_text = (
        "I have so many assignments and deadlines. "
        "I feel overwhelmed and cannot concentrate."
    )

    result = predictor.predict(sample_text)

    print("\n========== PREDICTION RESULTS ==========")

    print("Input:", result["text"])

    print(
        "Logistic Regression:",
        result["logistic_regression"]["prediction"]
    )

    print(
        "Linear SVM:",
        result["linear_svm"]["prediction"]
    )

    print(
        "RoBERTa V2:",
        result["roberta"]["prediction"]
    )

    print(
        "RoBERTa Confidence:",
        result["roberta"]["confidence"]
    )

    print(
        "No Stress Probability:",
        result["roberta"]["no_stress_probability"]
    )

    print(
        "Stress Probability:",
        result["roberta"]["stress_probability"]
    )

    print("\n========== TEST COMPLETED ==========")
