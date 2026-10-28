"""Live review and saved evaluation views."""

import html
import importlib
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from sentix import charts, dashboard
from sentix.dataset import badge
from sentix.deep import DeepPredictor
from sentix.mascot import presentation_mood, render_senti, unsure_threshold
from sentix.state import initial, keep_widget_state
from src.ui import load_reports

PAGES = {
    "Overview": "Sentiment, certainty and insights across the active dataset.",
    "Review Analyzer": "Read one review with the real model and see how sure it is.",
    "Review Explorer": "Search, filter and open individual reviews in the active dataset.",
    "Batch Analysis": "Score a CSV of reviews, map optional columns and download results.",
    "Explain": "Which words pushed the model toward positive or negative.",
    "Guess the Mood": "Test your reading against real rating-derived labels.",
    "Model & Evaluation": "What the model is, how it was measured and where it struggles.",
    "History": "Reviews you analyzed in this browser session.",
}

SAMPLES = {
    "😍 Love": "Absolutely love this product, works perfectly and arrived early!",
    "😡 Hate": "Terrible quality. It broke in two days. Total waste of money.",
    "😐 Meh": "It is okay. Does the job, nothing special.",
}


def set_sample(text: str) -> None:
    st.session_state.review = text
    st.session_state.sample_pending = True


def clear_review_result() -> None:
    """Discard the old prediction as soon as Streamlit receives an edit."""
    st.session_state.result = None


def clear_mode_results() -> None:
    clear_review_result()
    for key in ("analyzed_text", "bulk_result", "bulk_signature", "explanation"):
        st.session_state.pop(key, None)
    st.session_state.game_reveal = None
    st.session_state.game_score = [0, 0]


def live_review(predictor) -> None:
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.subheader("Every review has a feeling.")
        st.caption(
            "Try an example (written for demonstration; it still runs through the model), "
            "or paste a real review."
        )
        for col, (label, text) in zip(st.columns(3), SAMPLES.items()):
            col.button(label, on_click=set_sample, args=(text,), width="stretch")
        text = st.text_area(
            "Your product review",
            key="review",
            height=160,
            max_chars=20000,
            on_change=clear_review_result,
        )
        st.html(
            '<div class="editing-review-note">Editing review. Press Read the mood when ready.</div>'
        )
        requested = st.button("Read the mood", type="primary")
        if requested or st.session_state.pop("sample_pending", False):
            try:
                with st.spinner("Reading the mood…"):
                    started = time.perf_counter()
                    st.session_state.result = predictor.predict(text)
                    elapsed = (time.perf_counter() - started) * 1000
                st.session_state.analyzed_text = text
                st.session_state.result_meta = {
                    "elapsed_ms": elapsed,
                    "at": time.strftime("%H:%M:%S UTC", time.gmtime()),
                    "mode": st.session_state.get("analysis_mode", "Fast Mode"),
                }
                if st.session_state.result["status"] == "ok":
                    dashboard.record_history(
                        text, st.session_state.result, st.session_state.result_meta["mode"], elapsed
                    )
            except Exception:
                st.session_state.result = None
                st.warning("Senti couldn't read that just now. Try a shorter review.")
        result = st.session_state.get("result")
        if text != st.session_state.get("analyzed_text"):
            result = None
            st.session_state.result = None
            if text.strip():
                st.info("Review changed. Press Read the mood.")
        with st.container(key="live_verdict"):
            show_result(text, result)
    with right:
        result = st.session_state.get("result")
        valid = text.strip() and result and result["status"] == "ok"
        with st.container(key="live_mascot"):
            st.iframe(
                render_senti(
                    presentation_mood(result) if valid else "idle",
                    result["confidence"] if valid else 0,
                ),
                height=380,
            )
        st.caption("SENTI / your customer-listening companion")


def show_result(text: str, result: dict | None) -> None:
    """Keep model probabilities intact while making uncertainty explicit."""
    if text.strip() and result:
        if result["status"] == "ok":
            if result.get("warning"):
                st.warning(result["warning"])
            if result.get("route"):
                st.caption(
                    "Read by: int8 transformer"
                    if result["route"] == "transformer"
                    else "Read by: Fast Mode"
                )
            st.markdown(
                f'<span class="verdict {result["label"]}">{html.escape(badge(result["label"]))}</span>',
                unsafe_allow_html=True,
            )
            st.progress(
                result["confidence"],
                text=f"Predicted-class probability · {result['confidence']:.1%}",
            )
            st.altair_chart(charts.probabilities(result["probabilities"]), width="stretch")
            st.caption(
                "Probabilities: "
                + ", ".join(f"{k} {v:.1%}" for k, v in result["probabilities"].items())
                + "."
            )
            meta = st.session_state.get("result_meta")
            if meta:
                st.caption(
                    f"Analyzed at {meta['at']} · {meta['elapsed_ms']:.0f} ms · {meta['mode']} · "
                    "calibrated logistic regression on TF-IDF features"
                )
            if presentation_mood(result) == "unsure":
                st.info("Hmm, I'm not sure about this one. The model's label is tentative.")
                top_two = sorted(
                    result["probabilities"].items(), key=lambda item: item[1], reverse=True
                )[:2]
                st.write(
                    "Top two: "
                    + " · ".join(
                        f"{label.title()} {probability:.1%}" for label, probability in top_two
                    )
                )
            st.caption("Probabilities describe model uncertainty, not guaranteed correctness.")
        else:
            st.info(result.get("message", "Senti needs a little more text."))


@st.cache_data
def read_metrics(directory: str, metadata: dict, modified: int) -> dict:
    return load_reports(Path(directory), metadata)


def model_lab(predictor) -> None:
    st.subheader("The brain behind the face")
    st.write(
        "TF-IDF features → calibrated classifier → learned negative / neutral / positive labels."
    )
    directory = Path(
        os.environ.get("SENTIMENT_REPORTS_DIR", Path(__file__).resolve().parents[1] / "reports")
    )
    try:
        report = read_metrics(
            str(directory), predictor.metadata, (directory / "metrics.json").stat().st_mtime_ns
        )
    except Exception:
        report = None
    model_info(predictor, report)
    if report is None:
        st.info(
            "Senti's matching evaluation reports are unavailable. Run the notebook/evaluation workflow to generate metrics; no scores are invented here."
        )
        return
    try:
        evaluation(report, directory)
    except Exception:
        st.info("Some saved evaluation details could not be read. No scores are invented here.")


def model_info(predictor, report: dict | None) -> None:
    """Facts read from the artifact, its metadata and the hash-matched report only."""
    metadata = predictor.metadata
    vectorizer = predictor.model.named_steps.get("tfidf")
    ngrams = getattr(vectorizer, "ngram_range", None)
    with st.container(border=True):
        st.header("Model information")
        facts = {
            "Model": f"{metadata.get('selected_model', 'unknown')} classifier in one "
            "scikit-learn Pipeline (TF-IDF → classifier)",
            "Calibration": metadata.get("calibration", "not recorded"),
            "Labels": ", ".join(badge(label) for label in predictor.model.classes_),
            "Artifact": f"SHA-256 {metadata['artifact_sha256'][:12]}… · created "
            f"{str(metadata.get('created_utc', 'unknown'))[:10]}",
            "Features": "Word TF-IDF"
            + (f", n-grams {ngrams[0]}–{ngrams[1]}" if ngrams else "")
            + ". Inference runs locally; no review text leaves the server.",
            "Preprocessing": "HTML unescaped and tags removed, Unicode NFKC, lowercase, "
            "contractions expanded so negation survives (don't → do not), URLs and "
            "punctuation removed.",
            "Input rules": "1–20,000 characters with at least one two-letter word. Text with "
            "no familiar vocabulary returns unavailable, never a guessed label.",
            "Tentative threshold": f"Predictions below {unsure_threshold():.0%} predicted-class "
            "probability are flagged as tentative in the interface. Labels are unchanged.",
            "Library versions": ", ".join(
                f"{k} {v}" for k, v in metadata.get("versions", {}).items()
            ),
        }
        st.html(
            '<dl class="facts">'
            + "".join(
                f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in facts.items()
            )
            + "</dl>"
        )
        if not report:
            return
        audit = report.get("data_audit", {})
        sources = ", ".join(
            f"{s['filename']} ({s['rows']:,} rows)" for s in audit.get("sources", [])
        )
        st.markdown("**Training data provenance**")
        st.text(
            f"{sources}. After removing {audit.get('duplicate_rows_removed', 0):,} duplicates "
            f"and {audit.get('conflicting_rows_removed', 0):,} conflicting rows, "
            f"{audit.get('clean_rows', 0):,} reviews remained. Labels come from star ratings "
            "(low → negative, middle → neutral, high → positive)."
        )
        neutral = report["test"]["per_class"].get("neutral", {})
        st.markdown("**Known limitations**")
        st.text(
            f"Neutral reviews are hard for this model (test recall "
            f"{neutral.get('recall', 0):.1%}). Sarcasm, mixed opinions and non-English text "
            "are unreliable. Training data is mostly food reviews; the consumer-electronics "
            "slice is small. Rating-derived labels are themselves noisy."
        )


def evaluation(report: dict, directory: Path) -> None:
    metrics = report["test"]
    baseline = metrics.get("majority_baseline") or {}
    intervals = metrics.get("confidence_intervals") or {}
    with st.container(border=True):
        st.header("Held-out test evaluation")
        st.caption(
            f"Test split of {metrics['support']:,} reviews, scored once after model selection. "
            "Saved evaluation verified against the artifact hash."
        )
        a, b, c, d = st.columns(4)
        a.metric("Held-out accuracy", f"{metrics['accuracy']:.2%}", border=True)
        b.metric("Macro F1", f"{metrics['macro_f1']:.3f}", border=True)
        if baseline:
            c.metric(
                "Baseline accuracy",
                f"{baseline['accuracy']:.2%}",
                border=True,
                help="Always predicting the most common class. Accuracy alone flatters "
                "imbalanced data.",
            )
            d.metric("Baseline macro F1", f"{baseline['macro_f1']:.3f}", border=True)
        if intervals.get("accuracy"):
            st.caption(
                f"95% bootstrap intervals: accuracy [{intervals['accuracy']['lower']:.2%}, "
                f"{intervals['accuracy']['upper']:.2%}], macro F1 "
                f"[{intervals['macro_f1']['lower']:.3f}, {intervals['macro_f1']['upper']:.3f}] "
                f"({intervals.get('resamples', '?')} resamples; conditional on this fitted model)."
            )
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
            st.altair_chart(
                charts.confusion(metrics["confusion_matrix"], metrics["class_order"]),
                width="stretch",
            )
            st.dataframe(
                pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=metrics["class_order"],
                    columns=metrics["class_order"],
                )
            )
        with right:
            st.markdown("**Per-class results**")
            st.dataframe(
                pd.DataFrame(
                    {k: v for k, v in metrics["per_class"].items() if k in metrics["class_order"]}
                ).T
            )
            st.caption(
                "Neutral reviews are difficult for this model. Accuracy alone hides class imbalance."
            )
            sources = [
                {
                    "Source": name,
                    "Reviews": result["support"],
                    "Accuracy": result["accuracy"],
                    "Macro F1": result["macro_f1"],
                    "Low support": "yes" if result.get("low_support") else "no",
                }
                for name, result in report.get("per_source", {}).items()
            ]
            if sources:
                st.markdown("**By source** (within the same mixed test set)")
                st.dataframe(pd.DataFrame(sources), hide_index=True)
        figure = directory / "figures/calibration.png"
        if figure.exists():
            with st.expander("Reliability (calibration) curve"):
                st.image(str(figure))
                st.caption("How often predictions at each probability were right on the test set.")


def render(predictor) -> None:
    keep_widget_state()
    with st.sidebar:
        st.html(
            '<div class="brand"><span class="brand-mark" aria-hidden="true">✦</span>'
            "<div><strong>Sentix AI</strong><span>Product review intelligence</span></div></div>"
        )
        page = st.radio("Sections", list(PAGES), key="page", **initial("page", index=1))
        st.divider()
        mode = st.radio(
            "Analysis mode",
            ["Fast Mode", "Deep Mode"],
            key="analysis_mode",
            on_change=clear_mode_results,
            help="Deep Mode defers only when the production predicted-class probability is below 0.80. There is no separate neutral override.",
        )
        if mode == "Deep Mode":
            st.caption(
                "Deep Mode can improve neutral detection, but is slower and had lower overall accuracy in our test. First use downloads about 244 MB; Fast Mode is used if Deep Mode is unavailable."
            )
        # Filled after the page runs, since a page may replace the dataset.
        dataset_slot = st.empty()
        st.caption(
            "Deep Mode credits: [CardiffNLP Twitter-RoBERTa sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), CC BY 4.0. The int8 copy is a quantized, non-fine-tuned derivative."
        )
    selected = predictor
    progress_slot = st.empty()
    if mode == "Deep Mode":

        def progress(message, fraction):
            progress_slot.progress(fraction, text=message)

        try:
            selected = DeepPredictor(predictor, progress=progress)
        except Exception:
            st.warning("Deep Mode configuration is unavailable. Fast Mode is active.")
    st.html(
        f'<div class="page-head"><span class="eyebrow">Sentix AI · {html.escape(mode)}</span>'
        f"<h1>{html.escape(page)}</h1><p>{html.escape(PAGES[page])}</p></div>"
    )
    try:
        if page == "Overview":
            dashboard.overview(predictor)
        elif page == "Review Analyzer":
            live_review(selected)
        elif page == "Review Explorer":
            dashboard.explorer(predictor)
        elif page == "History":
            dashboard.history(predictor)
        elif page == "Model & Evaluation":
            st.caption("Saved production-model evaluation (Fast Mode).")
            model_lab(predictor)
        else:
            module = {"Batch Analysis": "bulk", "Explain": "explain", "Guess the Mood": "game"}[
                page
            ]
            if module == "explain" and mode == "Deep Mode":
                st.caption(
                    "This view explains the Fast Mode model only, not a deferred transformer result."
                )
            importlib.import_module(f"sentix.{module}").render(
                predictor if module == "explain" else selected
            )
    except Exception:
        st.info("Senti couldn't open this feature. Your live review is still available.")
    progress_slot.empty()
    meta = st.session_state.get("dataset_meta")
    frame = st.session_state.get("dataset")
    dataset_slot.caption(
        f"Active dataset: {meta['name']} ({len(frame):,} rows)"
        if meta and frame is not None
        else "Active dataset: none loaded"
    )
