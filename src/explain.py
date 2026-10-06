"""Linear score explanations, before sigmoid calibration; associations, not causation."""

import html
import re

import numpy as np
import pandas as pd

from src.preprocess import clean_text, validate_review


def linear_estimator(pipeline):
    classifier = pipeline.named_steps["clf"]
    if hasattr(classifier, "calibrated_classifiers_"):
        classifier = classifier.calibrated_classifiers_[0].estimator
    if hasattr(classifier, "estimator"):
        classifier = classifier.estimator
    return classifier


def explain_review(pipeline, text, label, limit=12):
    validate_review(text)
    classifier = linear_estimator(pipeline)
    if not hasattr(classifier, "coef_"):
        return pd.DataFrame(columns=["term", "contribution"])
    index = list(classifier.classes_).index(label)
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
    classifier = linear_estimator(pipeline)
    if not hasattr(classifier, "coef_"):
        return pd.DataFrame(columns=["term", "weight"])
    weights = classifier.coef_[list(classifier.classes_).index(label)]
    indices = np.unique(np.concatenate([np.argsort(weights)[:limit], np.argsort(weights)[-limit:]]))
    indices = indices[np.argsort(weights[indices])]
    return pd.DataFrame(
        {
            "term": pipeline.named_steps["tfidf"].get_feature_names_out()[indices],
            "weight": weights[indices],
        }
    )


def highlight_review(text, contributions):
    """Highlight normalized word features; escape all text before rendering HTML."""
    normalized = clean_text(text)
    terms = {}
    for row in contributions.itertuples():
        if row.term.startswith("char__"):
            continue
        term = row.term.removeprefix("word__")
        for word in term.split():
            terms[word] = terms.get(word, 0) + row.contribution
    parts = []
    for token in re.split(r"(\s+)", normalized):
        safe = html.escape(token)
        weight = terms.get(token, 0)
        if weight:
            color = "#cce9dc" if weight > 0 else "#f9d8d3"
            safe = f'<mark style="background:{color};color:#172b24;padding:2px">{safe}</mark>'
        parts.append(safe)
    return "".join(parts)
