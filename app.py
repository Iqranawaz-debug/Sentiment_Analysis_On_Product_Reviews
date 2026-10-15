"""Thin Sentix AI entry point. No training at startup."""

import os

import streamlit as st

from sentix.mascot import render_senti
from sentix.predictor import load_model
from sentix.theme import inject_theme
from sentix.views import render

st.set_page_config(page_title="Sentix AI · Meet Senti", page_icon="🌱", layout="wide")
inject_theme()


@st.cache_resource
def cached_model(directory: str):
    return load_model()


try:
    predictor = cached_model(os.environ.get("SENTIMENT_MODEL_DIR", "models"))
except Exception as error:
    st.iframe(render_senti("idle", 0, "My thinking cap needs a little help."), height=380)
    st.error("The model could not be loaded. Restore the matching model files and dependencies.")
    st.caption(str(error))
    st.code("python -m pip install -r requirements.txt")
    st.info("Senti never trains at startup. Your existing model remains unchanged.")
    st.stop()

render(predictor)
