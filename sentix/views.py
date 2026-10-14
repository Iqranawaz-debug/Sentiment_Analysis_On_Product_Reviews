"""Live review and saved evaluation views."""
import importlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from sentix.mascot import render_senti
from src.ui import load_reports

SAMPLES = {
    "😍 Love": "Absolutely love this product, works perfectly and arrived early!",
    "😡 Hate": "Terrible quality. It broke in two days. Total waste of money.",
    "😐 Meh": "It is okay. Does the job, nothing special.",
}


def set_sample(text: str) -> None:
    st.session_state.review = text


def live_review(predictor) -> None:
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.subheader("Every review has a feeling.")
        st.caption("Try a sample, or let Senti read your own.")
        for col, (label, text) in zip(st.columns(3), SAMPLES.items()):
            col.button(label, on_click=set_sample, args=(text,), width="stretch")
        text = st.text_area("Your product review", key="review", height=160, max_chars=20000)
        st.button("Read the mood", type="primary")
        if text != st.session_state.get("analyzed_text"):
            try:
                st.session_state.result = predictor.predict(text)
                st.session_state.analyzed_text = text
            except Exception:
                st.session_state.result = None
                st.warning("Senti couldn't read that just now. Try a shorter review.")
        result = st.session_state.get("result")
        if text.strip() and result:
            if result["status"] == "ok":
                st.markdown(f'<span class="verdict">{result["label"].title()}</span>', unsafe_allow_html=True)
                st.progress(result["confidence"], text=f'Predicted-class probability · {result["confidence"]:.1%}')
                st.bar_chart(pd.Series(result["probabilities"], name="Probability"), color="#65b998")
                if result["confidence"] < .60:
                    st.info("Senti is uncertain. Read the review alongside this prediction.")
                st.caption("Probabilities describe model uncertainty, not guaranteed correctness.")
            else:
                st.info(result.get("message", "Senti needs a little more text."))
    with right:
        result = st.session_state.get("result")
        valid = text.strip() and result and result["status"] == "ok"
        components.html(render_senti(result["label"] if valid else "idle", result["confidence"] if valid else 0), height=380)
        st.caption("SENTI / your customer-listening companion")


@st.cache_data
def read_metrics(directory: str, metadata: dict, modified: int) -> dict:
    return load_reports(Path(directory), metadata)


def model_lab(predictor) -> None:
    st.subheader("The brain behind the face")
    st.write("TF-IDF features → calibrated classifier → learned negative / neutral / positive labels.")
    directory = Path(os.environ.get("SENTIMENT_REPORTS_DIR", Path(__file__).resolve().parents[1] / "reports"))
    try:
        report = read_metrics(str(directory), predictor.metadata, (directory / "metrics.json").stat().st_mtime_ns)
        metrics = report["test"]
        a, b = st.columns(2)
        a.metric("Held-out accuracy", f'{metrics["accuracy"]:.2%}')
        b.metric("Macro F1", f'{metrics["macro_f1"]:.3f}')
        st.caption("Saved evaluation verified against the artifact hash. Rows = actual; columns = predicted.")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=metrics["class_order"], columns=metrics["class_order"]))
        st.dataframe(pd.DataFrame({k: v for k, v in metrics["per_class"].items() if k in metrics["class_order"]}).T)
        st.caption("Neutral reviews are difficult for this model. Accuracy alone hides class imbalance.")
    except Exception:
        st.info("Senti's matching evaluation reports are unavailable. Run the notebook/evaluation workflow to generate metrics; no scores are invented here.")


def render(predictor) -> None:
    tabs = st.tabs(["🐱 Live Review", "📦 Bulk Mode", "🔦 Why? (Explain)", "🎮 Guess the Mood", "📊 Model Lab"])
    with tabs[0]:
        live_review(predictor)
    for tab, module in zip(tabs[1:4], ("bulk", "explain", "game")):
        with tab:
            try:
                importlib.import_module(f"sentix.{module}").render(predictor)
            except Exception:
                st.info("Senti couldn't open this feature. Your live review is still available.")
    with tabs[4]:
        model_lab(predictor)
