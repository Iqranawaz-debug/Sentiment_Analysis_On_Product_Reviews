"""Signed positive-minus-negative evidence, not a causal explanation."""

import html
import re
from functools import lru_cache

import pandas as pd
import streamlit as st

from sentix.predictor import load_model
from src.explain import linear_estimator
from src.preprocessing import clean_text


@lru_cache(maxsize=1)
def default_predictor():
    return load_model()


def explain(text: str, predictor=None) -> list[tuple[str, float]]:
    """Prefer coefficient × TF-IDF score contrast; fall back to perturbation.

    N-gram contributions are divided among constituent words for display.
    Character features are excluded, so this is a partial score explanation.
    """
    predictor = predictor or default_predictor()
    if predictor.predict(text)["status"] != "ok":
        return []
    try:
        estimator = linear_estimator(predictor.model)
        classes = list(estimator.classes_)
        coefficients = estimator.coef_
        weights = (
            coefficients[classes.index("positive")] - coefficients[classes.index("negative")]
            if coefficients.shape[0] > 1
            else coefficients[0]
        )
        vectorizer = predictor.model.named_steps["tfidf"]
        features = vectorizer.transform([text]).tocoo()
        names = vectorizer.get_feature_names_out()
        words = {}
        for col, value in zip(features.col, features.data):
            name = names[col]
            if name.startswith(("char__", "structure__")):
                continue
            tokens = name.removeprefix("text__").removeprefix("word__").split()
            for token in tokens:
                words[token] = words.get(token, 0.0) + float(value * weights[col]) / len(tokens)
        return sorted(words.items(), key=lambda item: abs(item[1]), reverse=True)
    except Exception:
        return perturbation(text, predictor)


def perturbation(text: str, predictor) -> list[tuple[str, float]]:
    """Bound optional LIME or leave-one-word-out to the first 80 words."""
    text = " ".join(clean_text(text).split()[:80])
    try:
        from lime.lime_text import LimeTextExplainer

        def probabilities(texts):
            return predictor.model.predict_proba(list(texts))

        classes = list(predictor.model.classes_)
        positive, negative = classes.index("positive"), classes.index("negative")
        explanation = LimeTextExplainer(class_names=classes, random_state=42).explain_instance(
            text, probabilities, labels=(positive, negative), num_features=20, num_samples=128
        )
        values = dict(explanation.as_list(label=positive))
        for word, weight in explanation.as_list(label=negative):
            values[word] = values.get(word, 0) - weight
        return list(values.items())
    except Exception:
        tokens = text.split()
        words = list(dict.fromkeys(tokens))
        if not words:
            return []
        variants = [text] + [" ".join(w for w in tokens if w != word) for word in words]
        try:
            probabilities = predictor.model.predict_proba(variants)
            classes = list(predictor.model.classes_)
            contrasts = (
                probabilities[:, classes.index("positive")]
                - probabilities[:, classes.index("negative")]
            )
            return list(zip(words, (contrasts[0] - contrasts[1:]).astype(float)))
        except Exception:
            return []


def highlight(text: str, contributions: list[tuple[str, float]]) -> str:
    """Escape review HTML and scale color opacity by absolute contribution."""
    weights = dict(contributions)
    scale = max((abs(v) for v in weights.values()), default=1) or 1
    parts = []
    for token in re.split(r"(\w+)", clean_text(text)):
        value = weights.get(token, 0)
        safe = html.escape(token)
        if value:
            rgb = "83,184,130" if value > 0 else "232,111,111"
            safe = f'<mark style="background:rgba({rgb},{0.15 + 0.65 * abs(value) / scale:.3f});color:inherit;border-radius:4px;padding:2px" title="{value:+.4f}">{safe}</mark>'
        parts.append(safe)
    return '<div class="explanation">' + "".join(parts) + "</div>"


def render(predictor) -> None:
    st.subheader("Why did the Fast model read it this way?")
    st.caption(
        "For the Fast linear model this is an exact breakdown of its pre-calibration "
        "positive-minus-negative score into words. It shows which words moved the score, "
        "not whether the label is right, and it does not explain Deep Mode's transformer."
    )
    text = st.session_state.get("review", "")
    if not text.strip():
        st.info("Enter a review in Review Analyzer first.")
        return
    # Review text is untrusted: show it as plain text, never as Markdown or HTML.
    st.text(text)
    fast = predictor.predict(text)
    if fast["status"] == "ok":
        st.caption(
            f"Fast Mode label: {fast['label']} ({fast['confidence']:.1%}), chosen by "
            f"{fast.get('decision_rule', 'the most probable class')}."
        )
    if st.button("Explain this review"):
        try:
            st.session_state.explanation = (text, explain(text, predictor))
        except Exception:
            st.info("Senti can't explain this review yet. Try a shorter review.")
    saved = st.session_state.get("explanation")
    if saved and saved[0] == text:
        contributions = saved[1]
        if not contributions:
            st.info("No word explanation is available for this review.")
            return
        st.html(
            '<div class="legend"><span><i style="background:rgba(83,184,130,.7)"></i>Pushes '
            'toward positive</span><span><i style="background:rgba(232,111,111,.7)"></i>Pushes '
            "toward negative</span><span>Stronger colour means a bigger push; hover a word for "
            "its value</span></div>"
        )
        st.markdown(highlight(text, contributions), unsafe_allow_html=True)
        st.caption(
            "Green pushes toward positive; red toward negative. Linear explanations use pre-calibration positive-minus-negative scores, omit character features and intercepts, and divide n-grams among words. Text is normalized. Nonlinear fallback uses at most 80 words and probability perturbations."
        )
        for column, sign in zip(st.columns(2), (1, -1)):
            selected = sorted(
                (item for item in contributions if item[1] * sign > 0),
                key=lambda item: item[1] * sign,
                reverse=True,
            )[:5]
            column.dataframe(
                pd.DataFrame(selected, columns=["Word", "Contribution"]), hide_index=True
            )
