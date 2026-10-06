"""Stateless normalization shared by training and inference."""

import html
import re
import unicodedata

MAX_REVIEW_CHARS = 20_000


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(text)).lower()
    text = re.sub(r"<[^>]*>", " ", text)
    text = text.replace("’", "'")
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def validate_review(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty review.")
    if len(text) > MAX_REVIEW_CHARS:
        raise ValueError(f"Reviews must be at most {MAX_REVIEW_CHARS:,} characters.")
    if not re.search(r"[^\W\d_]{2,}", clean_text(text), re.UNICODE):
        raise ValueError("Include at least one word with two letters.")
    return text
