"""Validated inference and exports. Only load artifacts you trust."""

import importlib.metadata
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.data import LABELS, sha256_file
from src.decision import DecisionPolicy
from src.explain import explain_review, top_terms  # noqa: F401
from src.preprocessing import validate_review

MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_BATCH_ROWS = 1000


def load_artifact(directory):
    directory = Path(directory)
    model_path = directory / "pipeline.joblib"
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != 2:
            raise ValueError("Unsupported artifact schema. Retrain the model.")
        if sha256_file(model_path) != metadata["artifact_sha256"]:
            raise ValueError("Model checksum mismatch. Restore the matching artifact pair.")
        for package in ["scikit-learn", "numpy", "scipy", "joblib"]:
            if metadata["versions"][package] != importlib.metadata.version(package):
                raise ValueError(
                    f"{package} version mismatch. Install requirements.txt or retrain."
                )
        pipeline = joblib.load(model_path)
        if (
            not isinstance(pipeline, Pipeline)
            or set(pipeline.classes_) != set(LABELS)
            or "tfidf" not in pipeline.named_steps
            or not isinstance(
                pipeline.named_steps.get("clf"), (CalibratedClassifierCV, DecisionPolicy)
            )
        ):
            raise ValueError("Model does not match the three-class text pipeline contract.")
    except FileNotFoundError as error:
        raise ValueError(
            "Model files are missing. Run python -m src.train --config configs/default.yaml first."
        ) from error
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid model metadata. Restore or retrain the artifact pair.") from error
    return pipeline, metadata


def predict_reviews(pipeline, texts, threshold=0.60):
    if not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be between zero and one.")
    texts = list(texts)
    if not texts or len(texts) > MAX_BATCH_ROWS:
        raise ValueError(f"Provide 1 to {MAX_BATCH_ROWS:,} reviews.")
    for text in texts:
        validate_review(text)
    features = pipeline.named_steps["tfidf"].transform(texts)
    vectorizer = pipeline.named_steps["tfidf"]
    if hasattr(vectorizer, "named_steps") and "columns" in vectorizer.named_steps:
        features = vectorizer.named_steps["columns"].named_transformers_["text"].transform(texts)
    known = np.asarray(features.getnnz(axis=1)).ravel() > 0
    result = pd.DataFrame(
        {
            "review_text": texts,
            "sentiment": pipeline.predict(texts),
            "status": np.where(known, "ok", "no_known_terms"),
        }
    )
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(texts)
        for index, label in enumerate(pipeline.classes_):
            result[f"probability_{label}"] = probabilities[:, index]
        indices = np.array([list(pipeline.classes_).index(label) for label in result.sentiment])
        result["model_probability"] = probabilities[np.arange(len(texts)), indices]
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(texts)
        for index, label in enumerate(pipeline.classes_):
            result[f"margin_{label}"] = scores[:, index]
    result.loc[~known, "sentiment"] = "unavailable"
    numeric = result.select_dtypes(include="number").columns
    result.loc[~known, numeric] = np.nan
    result["low_confidence"] = (~known) | (
        result.get("model_probability", pd.Series(0, index=result.index)).fillna(0) < threshold
    )
    return result


def read_batch(content: bytes):
    if not content or len(content) > MAX_CSV_BYTES:
        raise ValueError("Upload a non-empty UTF-8 CSV no larger than 5 MB.")
    try:
        frame = pd.read_csv(
            io.BytesIO(content),
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=False,
            nrows=MAX_BATCH_ROWS + 1,
        )
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError("Could not read the file. Use a well-formed UTF-8 CSV.") from error
    if "review_text" not in frame.columns:
        raise ValueError("CSV must have a review_text column.")
    if not 1 <= len(frame) <= MAX_BATCH_ROWS:
        raise ValueError(f"CSV must contain 1 to {MAX_BATCH_ROWS:,} rows.")
    for index, text in enumerate(frame.review_text, start=2):
        try:
            validate_review(text)
        except ValueError as error:
            raise ValueError(f"CSV row {index}: {error}") from error
    return frame[["review_text"]]


def export_csv(frame):
    safe = frame.copy()
    # CSV quoting alone does not stop spreadsheet formula execution.
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].map(
            lambda value: (
                "'" + value
                if isinstance(value, str)
                and (
                    value.startswith(("\t", "\r", "\n"))
                    or value.lstrip().startswith(("=", "+", "-", "@"))
                )
                else value
            )
        )
    return safe.to_csv(index=False).encode("utf-8-sig")


def predict(pipeline, text, threshold=0.60):
    """Stable single-prediction schema with calibrated probability ordering."""
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("A probability-capable calibrated artifact is required.")
    row = predict_reviews(pipeline, [text], threshold).iloc[0]
    return {
        "label": row.sentiment,
        "probabilities": {
            label: None
            if pd.isna(row[f"probability_{label}"])
            else float(row[f"probability_{label}"])
            for label in LABELS
        },
        "low_confidence": bool(row.low_confidence),
    }
