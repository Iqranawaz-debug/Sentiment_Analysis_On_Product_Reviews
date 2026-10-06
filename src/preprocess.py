"""Negation-safe normalization and optional stopword ablation."""

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.preprocessing import MAX_REVIEW_CHARS, clean_text, validate_review

__all__ = ["MAX_REVIEW_CHARS", "clean_text", "validate_review", "without_stopwords"]
NEGATION = {"not", "no", "nor", "never", "n't"}
SAFE_STOPWORDS = ENGLISH_STOP_WORDS - NEGATION


def without_stopwords(text):
    return " ".join(word for word in clean_text(text).split() if word not in SAFE_STOPWORDS)
