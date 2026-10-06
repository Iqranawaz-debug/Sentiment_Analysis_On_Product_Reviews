"""Validated inference and exports. Only load artifacts you trust."""

import importlib.metadata
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import LABELS, sha256_file
from src.preprocessing import validate_review

MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_BATCH_ROWS = 1000


def load_artifact(directory):
    directory = Path(directory)
    model_path = directory / "pipeline.joblib"
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] not in (1, 2):
            raise ValueError("Unsupported artifact schema. Retrain the model.")
        if sha256_file(model_path) != metadata["artifact_sha256"]:
            raise ValueError("Model checksum mismatch. Restore the matching artifact pair.")
        for package in ["scikit-learn", "numpy", "scipy", "joblib"]:
            if metadata["versions"][package] != importlib.metadata.version(package):
                raise ValueError(
                    f"{package} version mismatch. Install requirements.txt or retrain."
                )
        pipeline = joblib.load(model_path)
        if set(pipeline.classes_) != set(LABELS) or "tfidf" not in pipeline.named_steps:
            raise ValueError("Model does not match the three-class text pipeline contract.")
    except FileNotFoundError as error:
        raise ValueError(
            "Model files are missing. Run python -m src.train --config configs/default.yaml first."
        ) from error
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid model metadata. Restore or retrain the artifact pair.") from error
    return pipeline, metadata


def predict_reviews(pipeline, texts):
    texts = list(texts)
    if not texts or len(texts) > MAX_BATCH_ROWS:
        raise ValueError(f"Provide 1 to {MAX_BATCH_ROWS:,} reviews.")
    for text in texts:
        validate_review(text)
    features = pipeline.named_steps["tfidf"].transform(texts)
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
        result["model_probability"] = probabilities.max(axis=1)
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(texts)
        for index, label in enumerate(pipeline.classes_):
            result[f"margin_{label}"] = scores[:, index]
    result.loc[~known, "sentiment"] = "unavailable"
    numeric = result.select_dtypes(include="number").columns
    result.loc[~known, numeric] = np.nan
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


def explain_review(pipeline, text, label, limit=12):
    validate_review(text)
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        return pd.DataFrame()
    index = list(pipeline.classes_).index(label)
    vectorizer = pipeline.named_steps["tfidf"]
    features = vectorizer.transform([text]).tocoo()
    contributions = features.data * classifier.coef_[index, features.col]
    order = np.argsort(np.abs(contributions))[::-1][:limit]
    return pd.DataFrame(
        {
            "term": vectorizer.get_feature_names_out()[features.col[order]],
            "contribution": contributions[order],
        }
    )


def top_terms(pipeline, label, limit=15):
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        return pd.DataFrame()
    weights = classifier.coef_[list(pipeline.classes_).index(label)]
    indices = np.argsort(weights)[-limit:][::-1]
    return pd.DataFrame(
        {
            "term": pipeline.named_steps["tfidf"].get_feature_names_out()[indices],
            "weight": weights[indices],
        }
    )
