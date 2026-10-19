"""Reviewable heuristic labels for a seeded random neutral-error sample."""

import re

from src.preprocess import clean_text

POSITIVE = {"good", "great", "excellent", "love", "lovely", "perfect", "delicious"}
NEGATIVE = {"bad", "awful", "terrible", "hate", "broken", "disappointed", "poor"}


def category(text):
    words = clean_text(text).split()
    tokens = set(words)
    if (tokens & POSITIVE and tokens & NEGATIVE) or re.search(
        r"\b(but|however|although|yet)\b", clean_text(text)
    ):
        return "mixed sentiment"
    if len(words) < 10:
        return "short text"
    if bool(tokens & POSITIVE) != bool(tokens & NEGATIVE):
        return "star/text mismatch"
    return "needs manual review"


def neutral_error_sample(rows, seed=42, limit=50):
    errors = rows[(rows.true_label == "neutral") & (rows.predicted_label != "neutral")]
    sampled = errors.sample(n=min(limit, len(errors)), random_state=seed).copy()
    if "star_rating" not in sampled:
        sampled["star_rating"] = float("nan")
    sampled["category"] = sampled.review_text.map(category)
    return sampled[
        [
            "review_text",
            "true_label",
            "predicted_label",
            "confidence",
            "star_rating",
            "source",
            "category",
        ]
    ]
