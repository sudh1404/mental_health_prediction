
from pathlib import Path

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_DIR = PROJECT_ROOT / "models" / "baseline_stress"
ROBERTA_DIR = PROJECT_ROOT / "models" / "roberta_stress"


class StressPredictor:

    def __init__(self):
        print("Loading Logistic Regression...")
        self.lr_model = joblib.load(
            BASELINE_DIR / "tfidf_logistic_stress.joblib"
        )

        print("Loading Linear SVM...")
        self.svm_model = joblib.load(
            BASELINE_DIR / "tfidf_linear_svm_stress.joblib"
        )

        print("Loading RoBERTa tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(ROBERTA_DIR),
            local_files_only=True
        )

        print("Loading trained RoBERTa...")
        self.roberta_model = (
            AutoModelForSequenceClassification.from_pretrained(
                str(ROBERTA_DIR),
                local_files_only=True
            )
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.roberta_model.to(self.device)
        self.roberta_model.eval()

        print("All three models loaded successfully!")

    def predict(self, text):

        if not text or not text.strip():
            raise ValueError("Please enter a social media post.")

        text = text.strip()

        # Traditional ML models
        lr_prediction = int(self.lr_model.predict([text])[0])
        svm_prediction = int(self.svm_model.predict([text])[0])

        # RoBERTa
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

        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

        roberta_prediction = int(
            torch.argmax(probabilities).item()
        )

        labels = {
            0: "No Stress",
            1: "Stress"
        }

        return {
            "text": text,

            "logistic_regression": {
                "prediction": labels[lr_prediction]
            },

            "linear_svm": {
                "prediction": labels[svm_prediction]
            },

            "roberta": {
                "prediction": labels[roberta_prediction],
                "confidence": round(
                    float(probabilities[roberta_prediction].item()),
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
        }


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
        "RoBERTa:",
        result["roberta"]["prediction"]
    )

    print(
        "RoBERTa Confidence:",
        result["roberta"]["confidence"]
    )

    print(
        "Stress Probability:",
        result["roberta"]["stress_probability"]
    )
