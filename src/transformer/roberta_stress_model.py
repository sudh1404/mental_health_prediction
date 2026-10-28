
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Locate the model relative to this script, regardless of terminal location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "roberta_stress"

LABELS = {
    0: "No Stress",
    1: "Stress"
}


def load_roberta_model():
    """Load the locally saved RoBERTa tokenizer and model."""

    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Model config not found in: {MODEL_DIR}"
        )

    if not (MODEL_DIR / "model.safetensors").exists():
        raise FileNotFoundError(
            f"Model weights not found in: {MODEL_DIR}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    model.eval()

    return tokenizer, model


def predict_stress(text, tokenizer, model):
    """Predict a stress-related text label and model confidence."""

    if not text.strip():
        raise ValueError("Please enter some text.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_class = int(torch.argmax(probabilities).item())

    return {
        "text": text,
        "prediction": LABELS[predicted_class],
        "confidence": round(
            float(probabilities[predicted_class].item()),
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


if __name__ == "__main__":

    print("Loading trained RoBERTa model...")

    tokenizer, model = load_roberta_model()

    print("Model loaded successfully!\n")

    sample_text = (
        "I have so many assignments and deadlines. "
        "I feel overwhelmed and cannot concentrate."
    )

    result = predict_stress(
        sample_text,
        tokenizer,
        model
    )

    print("=" * 50)
    print("ROBERTA STRESS PREDICTION")
    print("=" * 50)

    print("Text:", result["text"])
    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"])
    print("No Stress Probability:", result["no_stress_probability"])
    print("Stress Probability:", result["stress_probability"])
