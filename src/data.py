"""Load documented source schemas and audit data before splitting."""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import clean_text, validate_review

LABELS = ["negative", "neutral", "positive"]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(paths):
    frames, sources = [], []
    for path in paths:
        path = Path(path)
        columns = pd.read_csv(path, nrows=0).columns
        schemas = [
            ("reviews.text", "reviews.rating"),
            ("Text", "Score"),
            ("review_text", "score"),
            ("review_text", "sentiment"),
        ]
        schema = next((pair for pair in schemas if set(pair) <= set(columns)), None)
        if schema is None:
            raise ValueError(f"Unsupported training schema in {path.name}.")
        frame = pd.read_csv(path, usecols=list(schema)).rename(
            columns={schema[0]: "review_text", schema[1]: "label"}
        )
        frame["source"] = {"Text": "food", "reviews.text": "consumer-electronics"}.get(
            schema[0], path.stem
        )
        if schema[1] != "sentiment":
            ratings = pd.to_numeric(frame.label, errors="coerce")
            if (~ratings.dropna().isin([1, 2, 3, 4, 5])).any():
                raise ValueError(f"Ratings in {path.name} must be integers 1 through 5.")
            invalid = frame.label.notna() & ratings.isna()
            if invalid.any():
                raise ValueError(f"Non-numeric ratings in {path.name}.")
            frame["label"] = ratings.map(
                {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
            )
        else:
            frame["label"] = frame.label.astype("string").str.strip().str.lower()
            if not frame.label.dropna().isin(LABELS).all():
                raise ValueError(f"Unexpected sentiment labels in {path.name}.")
        sources.append({"filename": path.name, "rows": len(frame), "sha256": sha256_file(path)})
        frames.append(frame)
    if not frames:
        raise ValueError("Provide at least one training CSV.")
    return prepare_data(pd.concat(frames, ignore_index=True), sources)


def prepare_data(frame, sources=None):
    frame = frame.copy()
    if "source" not in frame:
        frame["source"] = "unspecified"
    audit = {"sources": sources or [], "input_rows": len(frame)}
    missing = frame[["review_text", "label"]].isna().any(axis=1)
    audit["missing_rows_removed"] = int(missing.sum())
    frame = frame.loc[~missing].copy()
    if not frame.label.isin(LABELS).all():
        raise ValueError("Labels must be negative, neutral, or positive.")

    def valid(value):
        try:
            validate_review(value)
            return True
        except ValueError:
            return False

    valid_mask = frame.review_text.map(valid)
    audit["invalid_text_rows_removed"] = int((~valid_mask).sum())
    frame = frame.loc[valid_mask].copy()
    frame["normalized"] = frame.review_text.map(clean_text)
    # Remove all conflicting groups instead of arbitrarily choosing a label.
    conflicts = frame.groupby("normalized").label.transform("nunique") > 1
    audit["conflicting_rows_removed"] = int(conflicts.sum())
    frame = frame.loc[~conflicts].copy()
    duplicates = frame.duplicated("normalized")
    audit["duplicate_rows_removed"] = int(duplicates.sum())
    frame = frame.loc[~duplicates].reset_index(drop=True)
    if set(frame.label) != set(LABELS):
        raise ValueError("Cleaned data must contain all three sentiment classes.")
    audit["clean_rows"] = len(frame)
    audit["class_distribution"] = frame.label.value_counts().sort_index().to_dict()
    audit["text_length_characters"] = frame.review_text.str.len().describe().to_dict()
    return frame, audit


def split_data(frame, seed=42, max_samples=None):
    """Input must be globally deduplicated; optional sampling retains imbalance."""
    if frame.normalized.duplicated().any():
        raise ValueError("Deduplicate before splitting.")
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive.")
        if max_samples < len(frame):
            frame, _ = train_test_split(
                frame, train_size=max_samples, stratify=frame.label, random_state=seed
            )
    train, rest = train_test_split(frame, test_size=0.30, stratify=frame.label, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=seed)
    return tuple(part.reset_index(drop=True) for part in (train, val, test))
