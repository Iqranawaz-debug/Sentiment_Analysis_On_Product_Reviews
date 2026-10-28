"""Session-state helpers shared by pages. Everything here lives in server memory only."""

import streamlit as st

FILTER_KEYS = (
    "ex_query", "ex_labels", "ex_conf", "ex_rating", "ex_products", "ex_dates",
    "ex_sort", "ex_desc", "ex_size", "ex_page", "ex_detail",
)  # fmt: skip
# Streamlit drops a widget's state on runs where the widget is not drawn. Re-assigning
# these before any widget renders keeps them when the user switches pages.
PERSISTENT = ("page", "analysis_mode", "review", "hist_query", "ov_freq", "ov_metric", *FILTER_KEYS)


def keep_widget_state() -> None:
    for key in PERSISTENT:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def initial(key: str, **defaults) -> dict:
    """Pass a widget default only on first draw, avoiding Streamlit's double-value warning."""
    return {} if key in st.session_state else defaults


def go(page: str) -> None:
    st.session_state.page = page


def reset_filters() -> None:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


def set_dataset(frame, meta: dict) -> None:
    """Replace the active dataset; stale filter values would not match its options."""
    st.session_state.dataset = frame
    st.session_state.dataset_meta = meta
    reset_filters()


def clear_dataset() -> None:
    st.session_state.pop("dataset", None)
    st.session_state.pop("dataset_meta", None)
    reset_filters()
