"""Bounded CSV analysis with honest denominators and safe spreadsheet export."""

import io
import os

import pandas as pd
import streamlit as st

from sentix import dataset as ds
from sentix.dataset import mood_score  # noqa: F401  (re-exported for existing callers)
from sentix.mascot import render_senti
from sentix.predictor import LABELS
from sentix.state import go, set_dataset
from src.predict import export_csv

MAX_BYTES = 10 * 1024 * 1024
NONE = "(none)"


@st.cache_data
def read_csv(content: bytes, limit: int) -> tuple[pd.DataFrame, bool]:
    if not content.strip():
        raise ValueError("This file is empty. Add a header row and at least one review.")
    if len(content) > MAX_BYTES:
        raise ValueError("Please use a CSV smaller than 10 MB.")
    try:
        frame = pd.read_csv(
            io.BytesIO(content),
            dtype=str,
            keep_default_na=False,
            nrows=limit + 1,
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError as error:
        raise ValueError("This file is not UTF-8 text. Re-save it as UTF-8 CSV.") from error
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError("This file is not a well-formed CSV with a header row.") from error
    return frame.head(limit), len(frame) > limit


def analyze(predictor, texts: list[str], progress=None) -> pd.DataFrame:
    """One output row per input row, in order; unreadable rows keep their reason."""
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
                    "message": result.get("message", ""),
                    "route": result.get("route", "fast"),
                    "warning": result.get("warning", ""),
                    **{f"probability_{k}": v for k, v in result["probabilities"].items()},
                }
            )
        if progress:
            progress(min((start + 128) / len(texts), 1.0))
    return pd.DataFrame(rows)


def upload_limit_mb() -> int:
    """The server's upload cap applies before our own 10 MB check, so report the smaller."""
    try:
        return min(int(st.get_option("server.maxUploadSize")), MAX_BYTES // (1024 * 1024))
    except Exception:
        return MAX_BYTES // (1024 * 1024)


def column_mapping(names: list[str]) -> dict:
    """Optional metadata columns. Nothing is assumed when the user leaves (none)."""
    titles = {
        "rating": "Rating column",
        "date": "Date column",
        "product": "Product or category column",
        "row_id": "Row ID column",
    }
    helps = {
        "rating": "Numeric star rating. Enables rating vs predicted-sentiment views.",
        "date": "ISO dates or Unix timestamps; read as UTC when no zone is given. Enables trends.",
        "product": "Enables product filtering in Review Explorer.",
        "row_id": "Your own identifier, kept in exports so results match input rows.",
    }
    mapping = {}
    columns = st.columns(4)
    signature = abs(hash(tuple(names)))
    for column, kind in zip(columns, titles):
        guess = ds.guess_column(names, kind)
        options = [NONE, *names]
        chosen = column.selectbox(
            titles[kind],
            options,
            index=options.index(guess) if guess else 0,
            key=f"map_{kind}_{signature}",
            help=helps[kind],
        )
        if chosen != NONE:
            mapping[kind] = chosen
    return mapping


def render(predictor) -> None:
    st.subheader("A whole crowd of opinions")
    limit = max(1, int(os.environ.get("SENTIX_MAX_ROWS", "5000")))
    st.caption(
        f"UTF-8 CSV · up to {limit:,} rows · {upload_limit_mb()} MB. Every row is kept: "
        "unreadable reviews are marked unavailable with a reason and excluded from shares."
    )
    st.download_button(
        "Download example CSV",
        b'review_text\n"Great quality."\n"Terrible waste of money."\n"It is average."\n',
        "example_reviews.csv",
        "text/csv",
    )
    uploaded = st.file_uploader("Review CSV", type="csv")
    if uploaded is None:
        meta = st.session_state.get("dataset_meta")
        if meta:
            st.info(f"Active dataset: {meta['name']}. Upload a CSV to replace it.")
            st.button("View it in Overview", on_click=go, args=("Overview",))
        return
    try:
        content = uploaded.getvalue()
        frame, truncated = read_csv(content, limit)
    except ValueError as error:
        st.warning(f"Senti couldn't read this file. {error}")
        return
    except Exception:
        st.warning(
            "Senti couldn't read this batch. Use a UTF-8 CSV under 10 MB with a review-text column."
        )
        return
    try:
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
        with st.expander("Optional columns and preview", expanded=False):
            mapping = column_mapping(names)
            st.dataframe(frame.head(5), hide_index=True)
            st.caption(f"First 5 of {len(frame):,} rows, shown exactly as read.")
        cells = frame[column].str.strip()
        blank = int(cells.eq("").sum())
        repeated = int(cells[cells.ne("")].str.lower().duplicated().sum())
        st.caption(
            f"{len(frame):,} rows ready · {blank:,} empty review cells will be reported as "
            "unavailable, not dropped"
            + (
                f" · {repeated:,} rows repeat an earlier review; all are kept and scored"
                if repeated
                else ""
            )
            + "."
        )
        signature = (content, column, limit)
        if st.button("Analyze CSV", type="primary"):
            progress = st.progress(0.0, text="Scoring reviews…")
            st.session_state.bulk_result = analyze(
                predictor, frame[column].tolist(), progress.progress
            )
            st.session_state.bulk_signature = signature
            progress.empty()
            st.toast(f"Analysis complete: {len(frame):,} rows.", icon=":material/check_circle:")
        if st.session_state.get("bulk_signature") != signature:
            return
        result = st.session_state.bulk_result
        meta = st.session_state.get("dataset_meta") or {}
        if meta.get("signature") != signature or meta.get("mapping") != mapping:
            set_dataset(
                ds.build_dataset(frame, result, mapping),
                {
                    "name": uploaded.name,
                    "kind": "upload",
                    "mode": st.session_state.get("analysis_mode", "Fast Mode"),
                    "signature": signature,
                    "mapping": mapping,
                    "note": f"Review text from column “{column}”."
                    + (f" Truncated to the first {limit:,} rows." if truncated else ""),
                },
            )
        fallbacks = (
            result["warning"].ne("")
            if "warning" in result
            else pd.Series(False, index=result.index)
        )
        if fallbacks.any():
            st.warning(
                f"{int(fallbacks.sum())} reviews used Fast Mode because Deep Mode was unavailable. {result.loc[fallbacks, 'warning'].iloc[0]}"
            )
        counts = (
            result[result.status == "ok"].sentiment.value_counts().reindex(LABELS, fill_value=0)
        )
        analyzed = int(counts.sum())
        st.success(
            f"Finished: {len(result):,} of {len(frame):,} input rows processed. "
            f"{analyzed:,} analyzed + {len(result) - analyzed:,} unavailable "
            f"= {len(result):,}."
            + (f" Rows after the first {limit:,} were not analyzed." if truncated else "")
        )
        score = mood_score(result)
        a, b = st.columns(2)
        a.button(
            "Open in Overview", on_click=go, args=("Overview",), type="primary", width="stretch"
        )
        b.button("Explore rows", on_click=go, args=("Review Explorer",), width="stretch")
        st.metric(
            "Product Mood Score",
            "Unavailable" if score is None else f"{score:.1f} / 100",
            help="50 × (1 + positive share − negative share), among valid predictions only.",
        )
        st.caption(f"{(result.status != 'ok').sum()} unavailable rows; {len(result)} total rows.")
        st.bar_chart(counts.rename("Reviews"), color="#137C73")
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
            "Download results as CSV",
            export_csv(st.session_state.dataset),
            "sentix_results.csv",
            "text/csv",
        )
    except Exception:
        st.warning(
            "Senti couldn't read this batch. Use a UTF-8 CSV under 10 MB with a review-text column."
        )
