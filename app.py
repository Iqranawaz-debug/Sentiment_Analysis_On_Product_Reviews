"""Streamlit entry point; training is an explicit offline command."""

import logging
import os
from pathlib import Path

import streamlit as st

from src.predict import load_artifact
from src.ui import render

st.set_page_config(page_title="Review Sentiment Lab", page_icon="💬", layout="wide")
ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("SENTIMENT_MODEL_DIR", str(ROOT / "models")))


@st.cache_resource
def cached_model(directory, model_modified, metadata_modified):
    return load_artifact(directory)


st.title("Review Sentiment Lab")
st.caption("Explore customer feedback • Compare evidence • Understand predictions")
try:
    model_file = MODEL_DIR / "pipeline.joblib"
    metadata_file = MODEL_DIR / "metadata.json"
    model, metadata = cached_model(
        str(MODEL_DIR), model_file.stat().st_mtime_ns, metadata_file.stat().st_mtime_ns
    )
except Exception:
    logging.getLogger(__name__).exception("Model initialization failed")
    st.error(
        "The model could not be loaded. Check the artifact files and installed dependency versions."
    )
    st.code("python -m pip install -r requirements.txt")
    st.code(
        "python -m src.train --config configs/default.yaml"
    )
    st.info(
        "The app never trains during startup. See README.md for setup and recovery instructions."
    )
    st.stop()

render(model, metadata)
