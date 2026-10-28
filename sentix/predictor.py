"""Adapter preserving the validated artifact's preprocessing and calibrated probabilities."""

import json
import os
from pathlib import Path

import numpy as np

from src.explain import linear_estimator
from src.predict import load_artifact, predict_reviews
from src.preprocessing import validate_review

LABELS = ("negative", "neutral", "positive")
ROOT = Path(__file__).resolve().parents[1]


def load_policy(directory: Path, artifact_sha256: str) -> dict | None:
    """The class-prior decision rule from scripts/tune_decision_policy.py.

    It is used only when it was tuned for this exact artifact and adopted under the
    protocol in reports/decision_policy.json; otherwise labels are the plain argmax.
    """
    try:
        policy = json.loads((directory / "decision_policy.json").read_text(encoding="utf-8"))
        weights = [float(policy["weights"][label]) for label in LABELS]
    except OSError, KeyError, TypeError, ValueError:
        return None
    if (
        policy.get("schema_version") != 1
        or policy.get("artifact_sha256") != artifact_sha256
        or not policy.get("adopted")
        or not all(np.isfinite(weights))
        or min(weights) <= 0
    ):
        return None
    return policy


class Predictor:
    """Unavailable input has null probabilities, never invented confidence.

    `confidence` is the calibrated probability of the returned label. With the
    class-prior rule active the returned label can differ from the most probable
    class; `top_probability` keeps the largest probability for routing and display.
    """

    mode = "Fast Mode"

    def __init__(self, model, metadata: dict, neutral_threshold: float = 0.60, policy=None):
        self.model = model
        self.metadata = metadata
        self.neutral_threshold = neutral_threshold
        self.policy = policy
        if not 0 <= neutral_threshold <= 1:
            raise ValueError("Neutral threshold must be between 0 and 1.")
        self.weights = (
            np.array([policy["weights"][label] for label in LABELS]) if policy else np.ones(3)
        )
        try:
            self.linear = linear_estimator(model)
            self.linear_order = [list(self.linear.classes_).index(label) for label in LABELS]
        except Exception:
            self.linear = None

    @property
    def decision_rule(self) -> str:
        return (
            f"class-prior rule (alpha {self.policy['alpha']:g}, tuned on validation)"
            if self.policy
            else "most probable class"
        )

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def raw_labels(self, texts: list[str]) -> list[str | None]:
        """Pre-calibration classifier choice, used only to flag disagreement."""
        if self.linear is None:
            return [None] * len(texts)
        features = self.model.named_steps["tfidf"].transform(texts)
        scores = self.linear.decision_function(features)[:, self.linear_order]
        return [LABELS[i] for i in scores.argmax(axis=1)]

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
            chunk = valid[start : start + 256]
            rows = predict_reviews(self.model, chunk)
            raw = self.raw_labels(chunk)
            for offset, row in enumerate(rows.itertuples()):
                target = indices[start + offset]
                if row.status != "ok":
                    results[target]["message"] = "Senti needs a few familiar English words."
                    continue
                values = np.array([float(getattr(row, f"probability_{k}")) for k in LABELS])
                if not np.all(np.isfinite(values)) or abs(values.sum() - 1) > 1e-6:
                    results[target]["message"] = "The model returned invalid scores."
                    continue
                probabilities = dict(zip(LABELS, values.tolist()))
                label = LABELS[int((values * self.weights).argmax())]
                top = LABELS[int(values.argmax())]
                notes = []
                if label != top:
                    notes.append("rule_changed_label")
                if raw[offset] is not None and raw[offset] != label:
                    notes.append("classifier_disagrees")
                results[target] = {
                    "label": label,
                    "confidence": probabilities[label],
                    "probabilities": probabilities,
                    "status": "ok",
                    "top_label": top,
                    "top_probability": probabilities[top],
                    "raw_label": raw[offset],
                    "uncertainty": notes,
                    "decision_rule": self.decision_rule,
                }
        return results


def load_model() -> Predictor:
    """Keep checksum and exact dependency validation in the existing loader.

    This artifact has a learned neutral class. Binary artifacts are deliberately
    rejected by that loader; a future binary adapter must mark low-confidence
    neutral as abstention, not fabricate a neutral probability.
    """
    directory = Path(os.environ.get("SENTIMENT_MODEL_DIR", ROOT / "models"))
    model, metadata = load_artifact(directory)
    return Predictor(model, metadata, policy=load_policy(directory, metadata["artifact_sha256"]))
