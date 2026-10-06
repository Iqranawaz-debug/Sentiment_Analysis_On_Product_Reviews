"""Six evidence-driven views; no fitting or training is imported by the app."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.explain import explain_review, highlight_review, top_terms
from src.predict import export_csv, predict_reviews, read_batch
from src.preprocess import MAX_REVIEW_CHARS

ROOT = Path(__file__).resolve().parents[1]


def single_review(model, threshold):
    st.subheader("What does this review say?")
    text = st.text_area(
        "Review",
        height=160,
        max_chars=MAX_REVIEW_CHARS,
        placeholder="The design is lovely, but it stopped working after a week.",
    )
    if st.button("Analyze review", type="primary"):
        try:
            result = predict_reviews(model, [text], threshold).iloc[0]
            if result.status != "ok":
                st.warning(
                    "No terms matched the model vocabulary. Try a more detailed English review."
                )
                return
            st.success(f"Predicted sentiment: {result.sentiment.title()}")
            if result.low_confidence:
                st.warning("Low confidence: this review needs human interpretation.")
            st.bar_chart(
                pd.Series(
                    {label: result[f"probability_{label}"] for label in model.classes_},
                    name="Calibrated probability",
                )
            )
            st.caption(
                "Calibrated probabilities describe uncertainty; they do not guarantee correctness."
            )
            explanation = explain_review(model, text, result.sentiment, limit=30)
            if not explanation.empty:
                st.markdown(highlight_review(text, explanation), unsafe_allow_html=True)
                st.caption(
                    "Normalized text: green words support this class score; red words oppose it. Character features appear in the table. Contributions explain the score before calibration, not probability changes or causality."
                )
                st.dataframe(explanation, hide_index=True)
        except ValueError as error:
            st.error(str(error))


def batch_reviews(model, threshold):
    st.subheader("Explore a batch of reviews")
    st.write("Upload a UTF-8 CSV with a `review_text` column. Extra columns are ignored.")
    st.caption(
        "Uploads are processed in memory. Avoid sensitive information on public deployments."
    )
    st.download_button(
        "Download example CSV",
        b'review_text\n"Great quality."\n"Not good at all."\n"It is average."\n',
        "example_reviews.csv",
        "text/csv",
    )
    uploaded = st.file_uploader("Review CSV", type=["csv"])
    if uploaded is not None:
        try:
            frame = read_batch(uploaded.getvalue())
            result = predict_reviews(model, frame.review_text, threshold)
            st.dataframe(result, hide_index=True)
            st.bar_chart(result.sentiment.value_counts().rename("Reviews"))
            st.download_button(
                "Download predictions", export_csv(result), "sentiment_predictions.csv", "text/csv"
            )
        except ValueError as error:
            st.error(str(error))


def load_reports(directory, metadata):
    """Fail closed on stale report/model pairs rather than showing unrelated metrics."""
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if metrics["artifact_sha256"] != metadata["artifact_sha256"]:
        raise ValueError("Reports belong to a different model. Restore the matching run.")
    return metrics


def performance(directory, metrics):
    st.subheader("Measured evidence")
    st.caption(
        "Families are compared using fixed settings in training-only CV. The winning family is tuned on validation. Test results are accessed after these choices."
    )
    comparison = pd.read_csv(directory / "comparison.csv")
    st.dataframe(
        comparison.drop(columns=["cv_folds", "cv_fold_scores"], errors="ignore"), hide_index=True
    )
    for column, metric in zip(st.columns(2), ("accuracy", "macro_f1")):
        column.metric(f"Test {metric.replace('_', ' ')}", f"{metrics['test'][metric]:.4f}")
    st.dataframe(pd.DataFrame(metrics["test"]["per_class"]).T)
    st.markdown("**Domain breakdown**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "source": source,
                    **{key: result[key] for key in ("accuracy", "macro_f1", "support")},
                }
                for source, result in metrics["per_source"].items()
            ]
        ),
        hide_index=True,
    )
    st.caption(
        "Source breakdown measures performance within the mixed-data test set; it is not an unseen-domain transfer experiment."
    )
    st.markdown("**Validation ablations**")
    st.dataframe(pd.read_csv(directory / "ablation.csv"), hide_index=True)
    st.image(str(directory / "figures/confusion_test.png"))
    st.image(str(directory / "figures/calibration.png"))
    if metrics.get("transformer"):
        st.markdown("**Optional transformer experiment**")
        st.json(metrics["transformer"])
    st.json(
        {
            "split_counts": metrics["split_counts"],
            "sample_cap": metrics["max_samples"],
            "data_audit": metrics["data_audit"],
        }
    )


def errors(directory):
    st.subheader("Most-confident mistakes")
    path = directory / "error_analysis.csv"
    if not path.exists():
        st.info(
            "Error samples are generated locally during training and are excluded from Git for review-text privacy."
        )
        return
    frame = pd.read_csv(path)
    if frame.empty:
        st.info("No misclassified test reviews in this run.")
        return
    actual = st.selectbox("True class", ["All", *sorted(frame.true_label.unique())])
    predicted = st.selectbox("Predicted class", ["All", *sorted(frame.predicted_label.unique())])
    confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0)
    source = st.selectbox("Source", ["All", *sorted(frame.source.unique())])
    if actual != "All":
        frame = frame[frame.true_label == actual]
    if predicted != "All":
        frame = frame[frame.predicted_label == predicted]
    if source != "All":
        frame = frame[frame.source == source]
    st.dataframe(frame[frame.confidence >= confidence], hide_index=True)
    st.caption(
        "This ranked error sample is deliberately biased toward confident failures; it is not a prevalence estimate."
    )


def render(model, metadata):
    threshold = metadata.get("low_confidence_threshold", 0.60)
    directory = Path(os.environ.get("SENTIMENT_REPORTS_DIR", str(ROOT / "reports")))
    tabs = st.tabs(["Predict", "Batch", "Model Comparison", "Error Explorer", "Insights", "About"])
    with tabs[0]:
        single_review(model, threshold)
    with tabs[1]:
        batch_reviews(model, threshold)
    try:
        metrics = load_reports(directory, metadata)
    except (OSError, ValueError, KeyError) as error:
        metrics = None
        with tabs[2]:
            st.info(f"Matching reports are unavailable. Run the offline training command. {error}")
    if metrics:
        with tabs[2]:
            try:
                performance(directory, metrics)
            except (OSError, ValueError, KeyError) as error:
                st.warning(f"Some generated report files are missing or invalid: {error}")
    with tabs[3]:
        if metrics:
            try:
                errors(directory)
            except (OSError, ValueError, AttributeError) as error:
                st.warning(f"Error samples could not be read: {error}")
        else:
            st.info("Matching reports are needed to explore errors.")
    with tabs[4]:
        label = st.selectbox("Sentiment class", list(model.classes_))
        terms = top_terms(model, label)
        if terms.empty:
            st.info("This baseline has no linear coefficients.")
        else:
            st.bar_chart(terms.set_index("term"))
            st.dataframe(terms, hide_index=True)
        st.caption(
            "Positive and negative coefficients for this class, before calibration. These are associations, not causal effects."
        )
    with tabs[5]:
        st.subheader("Review Sentiment Lab")
        st.write(
            "Amazon food and consumer-product reviews with rating-derived sentiment labels. English review triage for human exploration."
        )
        st.write(
            "Sarcasm, mixed sentiment, neutral ambiguity, source imbalance and domain shift remain limitations. Random splits do not isolate products, authors or time."
        )
        st.markdown(
            "[Model card on GitHub](https://github.com/Iqranawaz-debug/Sentiment_Analysis_On_Product_Reviews/blob/main/docs/MODEL_CARD.md)"
        )
        card = ROOT / "docs/MODEL_CARD.md"
        st.download_button(
            "Download model card",
            card.read_text(encoding="utf-8"),
            "MODEL_CARD.md",
            "text/markdown",
        )
        with st.expander("Model card"):
            st.markdown(card.read_text(encoding="utf-8"))
        st.caption(
            "The model trains offline and is cached for inference. Uploads are not deliberately persisted."
        )
