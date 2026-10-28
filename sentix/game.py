"""Guess the Mood: humans and model scored against known labels when available."""

import os
import random
from pathlib import Path

import pandas as pd
import streamlit as st

from sentix.dataset import display_text
from sentix.mascot import render_senti

FALLBACK = [
    "I love the bright display.",
    "It broke after one day.",
    "It is an ordinary product.",
    "Excellent quality and fast shipping.",
    "The packaging was damaged.",
    "It does the job.",
    "The battery lasts all week.",
    "A complete waste of money.",
    "Neither good nor bad.",
    "Delicious and fresh every time.",
    "The food tasted stale.",
    "An average purchase.",
    "Perfect fit and very comfortable.",
    "The size was completely wrong.",
    "It works as described.",
    "Fantastic value for the price.",
    "Customer service never replied.",
    "Some things work, some do not.",
    "I would happily buy this again.",
    "I returned it immediately.",
]


@st.cache_data
def load_reviews(path: str, modified: int) -> list[dict]:
    """Read a bounded sample of real held-out reviews, not their predictions."""
    try:
        frame = pd.read_csv(path, nrows=300, usecols=["review_text", "true_label"])
        frame = frame.dropna()
        frame = frame[frame.true_label.isin(["positive", "negative", "neutral"])]
        if not frame.empty:
            return frame.to_dict("records")
    except Exception:
        pass
    return [{"review_text": text, "true_label": None} for text in FALLBACK]


def score_round(guess: str, prediction: str, truth: str | None) -> tuple[int, int]:
    """Unknown truth means practice only; agreement never establishes truth."""
    if truth is None:
        return 0, 0
    return int(guess == truth), int(prediction == truth)


def new_round(reviews: list[dict]) -> None:
    st.session_state.game_review = random.choice(reviews)
    st.session_state.game_reveal = None


def render(predictor) -> None:
    st.subheader("Can you read the sentiment?")
    st.caption(
        "A practice game, separate from reporting. Guess whether a real review's star rating "
        "was positive, neutral or negative. It measures sentiment, not emotions."
    )
    directory = Path(
        os.environ.get("SENTIMENT_REPORTS_DIR", Path(__file__).resolve().parents[1] / "reports")
    )
    path = directory / "predictions.csv"
    reviews = load_reviews(str(path), path.stat().st_mtime_ns if path.exists() else 0)
    if "game_review" not in st.session_state:
        new_round(reviews)
    if "game_score" not in st.session_state:
        st.session_state.game_score = [0, 0]
    you, senti = st.session_state.game_score
    st.html(
        f'<div class="scoreboard" role="group" aria-label="Score"><div class="score"><b>{int(you)}'
        f'</b><span>You</span></div><div class="score"><b>{int(senti)}</b><span>Senti (the '
        "model)</span></div></div>"
    )
    st.caption(
        "One point each for matching the real rating-derived label. Both can score. Unlabeled examples are practice only."
    )
    review = st.session_state.game_review
    # Untrusted text: plain text only, never Markdown. Line-break tags shown as breaks.
    st.text(display_text(review["review_text"]))
    st.caption(
        "Real held-out review"
        if review["true_label"]
        else "Built-in example · no ground truth · unscored"
    )
    for column, (name, label) in zip(
        st.columns(3), [("Positive", "positive"), ("Neutral", "neutral"), ("Negative", "negative")]
    ):
        if column.button(name, disabled=st.session_state.game_reveal is not None):
            result = predictor.predict(review["review_text"])
            points = (
                score_round(label, result["label"], review["true_label"])
                if result["status"] == "ok"
                else (0, 0)
            )
            st.session_state.game_score = [you + points[0], senti + points[1]]
            st.session_state.game_reveal = (label, result, points)
            st.rerun()
    if st.session_state.game_reveal:
        guess, result, points = st.session_state.game_reveal
        if result.get("warning"):
            st.warning(result["warning"])
        st.write(
            f"You guessed **{guess}**. Senti predicted **{result['label']}**. Real label: **{review['true_label'] or 'unknown'}**."
        )
        message = (
            "You read that beautifully!"
            if points[0] > points[1]
            else "A point for my thinking cap!"
            if points[1] > points[0]
            else "We learn by listening. Another round?"
        )
        st.iframe(render_senti(result["label"], result["confidence"], message), height=380)
        if st.button("Next review"):
            new_round(reviews)
            st.rerun()
    if st.button("Reset score"):
        st.session_state.game_score = [0, 0]
        new_round(reviews)
        st.rerun()
