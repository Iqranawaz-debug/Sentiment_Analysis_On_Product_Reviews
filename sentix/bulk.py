"""Bounded CSV analysis with honest denominators and safe spreadsheet export."""

import io
import os

import pandas as pd
import streamlit as st

from sentix.mascot import render_senti
from sentix.predictor import LABELS
from src.predict import export_csv


@st.cache_data
def read_csv(content: bytes, limit: int) -> tuple[pd.DataFrame, bool]:
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("Please use a CSV smaller than 10 MB.")
    frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, nrows=limit + 1)
    return frame.head(limit), len(frame) > limit


def analyze(predictor, texts: list[str], progress=None) -> pd.DataFrame:
    rows = []
    for start in range(0, len(texts), 128):
        for text, result in zip(
            texts[start : start + 128], predictor.predict_batch(texts[start : start + 128])
        ):
            rows.append(
                {
                    "review_text": text,
                    "sentiment": result["label"],
                    "confidence": result["confidence"],
                    "status": result["status"],
                    **{f"probability_{k}": v for k, v in result["probabilities"].items()},
                }
            )
        if progress:
            progress(min((start + 128) / len(texts), 1.0))
    return pd.DataFrame(rows)


def mood_score(frame: pd.DataFrame) -> float | None:
    valid = frame[frame.status == "ok"]
    if valid.empty:
        return None
    return 50 * (
        1 + (valid.sentiment == "positive").mean() - (valid.sentiment == "negative").mean()
    )


def render(predictor) -> None:
    st.subheader("A whole crowd of opinions")
    limit = max(1, int(os.environ.get("SENTIX_MAX_ROWS", "5000")))
    st.caption(
        f"UTF-8 CSV · up to {limit:,} rows · 10 MB. Unavailable reviews are excluded from the mood score."
    )
    st.download_button(
        "Download example CSV",
        b'review_text\n"Great quality."\n"Terrible waste of money."\n"It is average."\n',
        "example_reviews.csv",
        "text/csv",
    )
    uploaded = st.file_uploader("Review CSV", type="csv")
    if uploaded is None:
        return
    try:
        content = uploaded.getvalue()
        frame, truncated = read_csv(content, limit)
        if frame.empty:
            st.info("Senti found no reviews in this file.")
            return
        if truncated:
            st.warning(f"Only the first {limit:,} rows will be analyzed.")
        names = list(frame.columns)
        preferred = next(
            (n for n in names if n.lower() in {"text", "review", "review_text", "reviews.text"}),
            names[0],
        )
        column = st.selectbox("Review-text column", names, index=names.index(preferred))
        signature = (content, column, limit)
        if st.button("Analyze CSV", type="primary"):
            progress = st.progress(0.0)
            st.session_state.bulk_result = analyze(
                predictor, frame[column].tolist(), progress.progress
            )
            st.session_state.bulk_signature = signature
            progress.empty()
        if st.session_state.get("bulk_signature") != signature:
            return
        result = st.session_state.bulk_result
        counts = (
            result[result.status == "ok"].sentiment.value_counts().reindex(LABELS, fill_value=0)
        )
        score = mood_score(result)
        st.metric(
            "Product Mood Score",
            "Unavailable" if score is None else f"{score:.1f} / 100",
            help="50 × (1 + positive share − negative share), among valid predictions only.",
        )
        st.caption(f"{(result.status != 'ok').sum()} unavailable rows; {len(result)} total rows.")
        st.bar_chart(counts.rename("Reviews"), color="#65b998")
        if counts.sum():
            exact = counts / counts.sum() * 20
            sizes = exact.astype(int)
            for label in (exact - sizes).sort_values(ascending=False).index[: 20 - sizes.sum()]:
                sizes[label] += 1
            st.write(
                " ".join(
                    face * int(sizes[label]) for label, face in zip(LABELS, ["😢", "😐", "😊"])
                )
            )
            message = (
                "Mostly happy customers! 🎉"
                if counts.positive > counts.sum() / 2
                else "Mixed voices — let's listen to the details."
            )
            st.iframe(render_senti("neutral", 0, message), height=380)
        st.dataframe(result, hide_index=True)
        for label in ("positive", "negative"):
            st.write(f"Top 5 most {label} reviews (model probability)")
            st.dataframe(
                result[result.status == "ok"].nlargest(5, f"probability_{label}")[
                    ["review_text", f"probability_{label}"]
                ],
                hide_index=True,
            )
        st.download_button(
            "Download results as CSV", export_csv(result), "sentix_results.csv", "text/csv"
        )
    except Exception:
        st.warning(
            "Senti couldn't read this batch. Use a UTF-8 CSV under 10 MB with a review-text column."
        )
