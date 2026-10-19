"""Live review and saved evaluation views."""

import importlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from sentix.deep import DeepPredictor
from sentix.mascot import presentation_mood, render_senti
from src.ui import load_reports

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
        st.caption("Try a sample, or let Senti read your own.")
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
                    st.session_state.result = predictor.predict(text)
                st.session_state.analyzed_text = text
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
                f'<span class="verdict">{result["label"].title()}</span>',
                unsafe_allow_html=True,
            )
            st.progress(
                result["confidence"],
                text=f"Predicted-class probability · {result['confidence']:.1%}",
            )
            st.bar_chart(pd.Series(result["probabilities"], name="Probability"), color="#65b998")
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
        metrics = report["test"]
        a, b = st.columns(2)
        a.metric("Held-out accuracy", f"{metrics['accuracy']:.2%}")
        b.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        st.caption(
            "Saved evaluation verified against the artifact hash. Rows = actual; columns = predicted."
        )
        st.dataframe(
            pd.DataFrame(
                metrics["confusion_matrix"],
                index=metrics["class_order"],
                columns=metrics["class_order"],
            )
        )
        st.dataframe(
            pd.DataFrame(
                {k: v for k, v in metrics["per_class"].items() if k in metrics["class_order"]}
            ).T
        )
        st.caption(
            "Neutral reviews are difficult for this model. Accuracy alone hides class imbalance."
        )
    except Exception:
        st.info(
            "Senti's matching evaluation reports are unavailable. Run the notebook/evaluation workflow to generate metrics; no scores are invented here."
        )


def render(predictor) -> None:
    mode = st.radio(
        "Analysis mode",
        ["Fast Mode", "Deep Mode"],
        horizontal=True,
        key="analysis_mode",
        on_change=clear_mode_results,
        help="Deep Mode defers only when the production predicted-class probability is below 0.80. There is no separate neutral override.",
    )
    selected = predictor
    progress_slot = st.empty()
    if mode == "Deep Mode":
        st.caption(
            "Deep Mode can improve neutral detection, but is slower and had lower overall accuracy in our test. First use downloads about 244 MB; Fast Mode is used if Deep Mode is unavailable."
        )

        def progress(message, fraction):
            progress_slot.progress(fraction, text=message)

        try:
            selected = DeepPredictor(predictor, progress=progress)
        except Exception:
            st.warning("Deep Mode configuration is unavailable. Fast Mode is active.")
    tabs = st.tabs(
        ["🐱 Live Review", "📦 Bulk Mode", "🔦 Why? (Explain)", "🎮 Guess the Mood", "📊 Model Lab"]
    )
    with tabs[0]:
        live_review(selected)
    for tab, module in zip(tabs[1:4], ("bulk", "explain", "game")):
        with tab:
            try:
                if module == "explain" and mode == "Deep Mode":
                    st.caption(
                        "This view explains the Fast Mode model only, not a deferred transformer result."
                    )
                importlib.import_module(f"sentix.{module}").render(
                    predictor if module == "explain" else selected
                )
            except Exception:
                st.info("Senti couldn't open this feature. Your live review is still available.")
    with tabs[4]:
        st.caption("Saved production-model evaluation (Fast Mode).")
        model_lab(predictor)
    progress_slot.empty()
    st.caption(
        "Deep Mode credits: [CardiffNLP Twitter-RoBERTa sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), CC BY 4.0. The int8 copy is a quantized, non-fine-tuned derivative."
    )
