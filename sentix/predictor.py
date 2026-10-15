"""Adapter preserving the validated artifact's preprocessing and decision policy."""

import os
from pathlib import Path

from src.predict import load_artifact, predict_reviews
from src.preprocessing import validate_review

LABELS = ("negative", "neutral", "positive")
ROOT = Path(__file__).resolve().parents[1]


class Predictor:
    """Unavailable input has null probabilities, never invented confidence."""

    def __init__(self, model, metadata: dict, neutral_threshold: float = 0.60):
        self.model = model
        self.metadata = metadata
        self.neutral_threshold = neutral_threshold
        if not 0 <= neutral_threshold <= 1:
            raise ValueError("Neutral threshold must be between 0 and 1.")

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Validate rows individually; batch valid rows using the existing pipeline."""
        results = []
        valid, indices = [], []
        for text in texts:
            result = {
                "label": "unavailable",
                "confidence": None,
                "probabilities": dict.fromkeys(LABELS),
                "status": "unavailable",
            }
            try:
                validate_review(text)
                indices.append(len(results))
                valid.append(text)
            except ValueError as error:
                result["message"] = str(error)
            results.append(result)
        for start in range(0, len(valid), 256):
            rows = predict_reviews(self.model, valid[start : start + 256])
            for offset, row in enumerate(rows.itertuples()):
                target = indices[start + offset]
                if row.status != "ok":
                    results[target]["message"] = "Senti needs a few familiar English words."
                    continue
                probabilities = {
                    label: float(getattr(row, f"probability_{label}")) for label in LABELS
                }
                label = str(row.sentiment).lower()
                results[target] = {
                    "label": label,
                    "confidence": probabilities[label],
                    "probabilities": probabilities,
                    "status": "ok",
                }
        return results


def load_model() -> Predictor:
    """Keep checksum and exact dependency validation in the existing loader.

    This artifact has a learned neutral class. Binary artifacts are deliberately
    rejected by that loader; a future binary adapter must mark low-confidence
    neutral as abstention, not fabricate a neutral probability.
    """
    model, metadata = load_artifact(Path(os.environ.get("SENTIMENT_MODEL_DIR", ROOT / "models")))
    return Predictor(model, metadata)
