"""Overview, Review Explorer and History over the active dataset.

The active dataset is either a scored CSV (Batch Analysis), the bundled sample, or this
session's analyzer history. It lives in Streamlit session memory only.
"""

import html

import pandas as pd
import streamlit as st

from sentix import charts
from sentix import dataset as ds
from sentix.bulk import analyze
from sentix.mascot import unsure_threshold
from sentix.state import go, initial, reset_filters, set_dataset
from src.predict import export_csv

HISTORY_LIMIT = 500
PAGE_SIZES = (10, 25, 50, 100)
SORTS = {
    "Row order": "row",
    "Predicted-class probability": "confidence",
    "Rating": "rating",
    "Date": "date",
}
SAMPLE_NOTE = (
    "Real Amazon reviews and star ratings from a tracked evaluation split "
    "(reports/approved_20261016/test.csv), re-scored live by the Fast Mode model. "
    "They may overlap the production model's training data, so this sample is never "
    "used to measure accuracy. It has no review dates."
)


def active() -> tuple[pd.DataFrame | None, dict | None]:
    return st.session_state.get("dataset"), st.session_state.get("dataset_meta")


@st.cache_data(show_spinner=False, max_entries=2)
def scored_sample(artifact_sha: str, _predictor) -> pd.DataFrame:
    """Cache by artifact hash so a replaced model is never served stale predictions."""
    source, mapping = ds.load_sample()
    return ds.build_dataset(source, analyze(_predictor, source.review_text.tolist()), mapping)


def load_sample_dataset(predictor) -> None:
    with st.spinner("Scoring 2,698 sample reviews with the Fast Mode model…"):
        frame = scored_sample(predictor.metadata["artifact_sha256"], predictor)
    set_dataset(
        frame,
        {"name": ds.SAMPLE_NAME, "kind": "sample", "mode": "Fast Mode", "note": SAMPLE_NOTE},
    )


def use_history() -> None:
    entries = st.session_state.get("history", [])
    if entries:
        set_dataset(
            ds.history_dataset(entries),
            {
                "name": f"This session's analyses ({len(entries)})",
                "kind": "history",
                "mode": "As recorded per review",
                "note": "Reviews you analyzed in Review Analyzer during this browser session.",
            },
        )
        go("Overview")


def record_history(text: str, result: dict, mode: str, elapsed_ms: float) -> None:
    entries = st.session_state.setdefault("history", [])
    entries.append(ds.history_entry(text, result, mode, elapsed_ms))
    del entries[:-HISTORY_LIMIT]


def cards(items: list[tuple[str, str, str, str]]) -> None:
    """KPI cards as one responsive grid. Values are numbers we format; text is escaped."""
    body = "".join(
        f'<div class="kpi {tone}" role="group" aria-label="{html.escape(title)}">'
        f'<div class="kpi-title">{html.escape(title)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'<div class="kpi-note">{html.escape(note)}</div></div>'
        for title, value, note, tone in items
    )
    st.html(f'<div class="kpi-grid">{body}</div>')


def insight_list(items: list[dict]) -> None:
    """Insights quote review words, so they are HTML-escaped rather than Markdown."""
    if not items:
        st.info("No insights yet: there are no analyzed reviews in this dataset.")
        return
    st.html(
        '<ul class="insights">'
        + "".join(
            f"<li><strong>{html.escape(i['title'])}</strong><span>{html.escape(i['detail'])}</span></li>"
            for i in items
        )
        + "</ul>"
    )


def display_frame(frame: pd.DataFrame, preview: int = 160) -> pd.DataFrame:
    out = pd.DataFrame({"Row": frame.row.to_numpy()})
    if "row_id" in frame:
        out["ID"] = frame.row_id.to_numpy()
    out["Sentiment"] = frame.sentiment.map(ds.badge).to_numpy()
    out["Probability"] = frame.confidence.to_numpy()
    if "rating" in frame:
        out["Rating"] = frame.rating.to_numpy()
    if "date" in frame:
        out["Date (UTC)"] = frame.date.dt.tz_convert(None).to_numpy()
    if "product" in frame:
        out["Product"] = frame["product"].to_numpy()
    out["Review"] = (
        frame.review_text.astype(str)
        .map(lambda t: t if len(t) <= preview else t[: preview - 1] + "…")
        .to_numpy()
    )
    return out


def table(frame: pd.DataFrame, **kwargs) -> None:
    st.dataframe(
        display_frame(frame),
        hide_index=True,
        column_config={
            "Probability": st.column_config.NumberColumn(
                format="percent", help="Calibrated predicted-class probability, not correctness."
            ),
            "Date (UTC)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Review": st.column_config.TextColumn(width="large"),
        },
        **kwargs,
    )


def context(meta: dict, frame: pd.DataFrame) -> None:
    mapped = [name for name in ("rating", "date", "product", "row_id") if name in frame]
    left, right = st.columns([4, 1.4], vertical_alignment="center")
    left.html(
        '<div class="context"><span class="context-label">Active dataset</span>'
        f"<strong>{html.escape(meta['name'])}</strong>"
        f"<span>{len(frame):,} rows · scored with {html.escape(meta['mode'])} · "
        f"optional columns: {html.escape(', '.join(mapped) or 'none mapped')}</span></div>"
    )
    with right:
        st.button("Change dataset", on_click=go, args=("Batch Analysis",), width="stretch")
    if meta.get("note"):
        st.caption(meta["note"])


def empty_state(predictor) -> None:
    st.html(
        '<div class="empty"><h2>No dataset loaded yet</h2><p>The overview summarizes a set of '
        "reviews scored by the model. Upload a CSV in Batch Analysis, load the bundled sample "
        "of real reviews, or use the reviews you analyzed in this session.</p></div>"
    )
    a, b, c = st.columns(3)
    if a.button("Load sample dataset", type="primary", width="stretch"):
        load_sample_dataset(predictor)
        st.rerun()
    b.button("Upload a CSV", on_click=go, args=("Batch Analysis",), width="stretch")
    entries = len(st.session_state.get("history", []))
    c.button(
        f"Use my {entries} analyzed reviews" if entries else "No analyzed reviews yet",
        on_click=use_history,
        disabled=not entries,
        width="stretch",
    )


def overview(predictor) -> None:
    frame, meta = active()
    if frame is None:
        empty_state(predictor)
        return
    context(meta, frame)
    low = unsure_threshold()
    stats = ds.kpis(frame, low)
    n = stats["analyzed"]
    shares = stats["shares"]
    cards(
        [
            (
                "Reviews analyzed",
                f"{n:,}",
                f"of {stats['total']:,} rows; {stats['unavailable']:,} unreadable rows excluded",
                "info",
            ),
            *[
                (
                    ds.badge(label),
                    "—" if shares[label] is None else f"{shares[label]:.1%}",
                    f"{stats['counts'][label]:,} of {n:,} analyzed reviews",
                    label,
                )
                for label in ("positive", "neutral", "negative")
            ],
        ]
    )
    score = stats["mood_score"]
    rating = stats["mean_rating"]
    cards(
        [
            (
                "Mean predicted-class probability",
                "—" if stats["mean_confidence"] is None else f"{stats['mean_confidence']:.1%}",
                "Calibrated model certainty, not accuracy",
                "info",
            ),
            (
                "Tentative predictions",
                f"{stats['low_confidence']:,}",
                f"Predicted-class probability below {low:.0%}",
                "info",
            ),
            (
                "Product Mood Score",
                "—" if score is None else f"{score:.1f} / 100",
                "50 × (1 + positive share − negative share)",
                "info",
            ),
            (
                "Average rating",
                "Not mapped"
                if "rating" not in frame
                else "—"
                if rating is None
                else f"{rating:.2f}",
                f"Mean of {stats['rated']:,} rated rows"
                if "rating" in frame
                else "Map a rating column in Batch Analysis",
                "info",
            ),
        ]
    )

    left, right = st.columns([3, 2], gap="large")
    with left, st.container(border=True):
        st.header("Sentiment distribution")
        table_ = ds.distribution(frame)
        if n:
            st.altair_chart(charts.distribution(table_), width="stretch")
            st.caption(
                "Shares of analyzed reviews: "
                + "; ".join(
                    f"{r.sentiment} {r.count:,} ({r.share:.1%})" for r in table_.itertuples()
                )
                + "."
            )
        else:
            st.info("No review in this dataset could be analyzed, so there is nothing to chart.")
    with right, st.container(border=True):
        st.header("Insights")
        insight_list(ds.insights(frame, low))

    with st.container(border=True):
        st.header("Sentiment over time")
        trend_section(frame)

    with st.container(border=True):
        st.header("Ratings and predicted sentiment")
        rating_section(frame)

    with st.container(border=True):
        st.header("Model certainty")
        if n:
            st.altair_chart(charts.confidence(frame), width="stretch")
            st.caption(
                f"Predicted-class probabilities for {n:,} analyzed reviews. "
                f"{stats['low_confidence']:,} fall below the {low:.0%} tentative threshold. "
                "Calibrated probabilities describe model uncertainty; they are not correctness."
            )
        else:
            st.info("No analyzed reviews.")

    with st.container(border=True):
        dated = "date" in frame and frame.date.notna().any()
        st.header("Most recent reviews" if dated else "First reviews in this dataset")
        rows = frame.sort_values("date", ascending=False) if dated else frame
        table(rows.head(10))
        st.button("Open Review Explorer", on_click=go, args=("Review Explorer",))


def trend_section(frame: pd.DataFrame) -> None:
    if "date" not in frame:
        st.info(
            "Trends need real review dates. Map a date column when uploading a CSV in Batch "
            "Analysis; dates are never invented."
        )
        return
    a, b = st.columns(2)
    choice = a.selectbox("Group by", ["Auto", "Day", "Week", "Month", "Year"], key="ov_freq")
    field = b.radio("Show", ["Count", "Share"], horizontal=True, key="ov_metric")
    freq = {"Auto": None, "Day": "D", "Week": "W", "Month": "M", "Year": "Y"}[choice]
    table_, used = ds.trend(frame, freq)
    analyzed = int((frame.status == "ok").sum())
    dated = int(((frame.status == "ok") & frame.date.notna()).sum())
    if table_.empty:
        st.info("None of the analyzed rows has a date that could be parsed.")
        return
    periods = table_.period.nunique()
    if periods > 400:
        st.warning(f"{periods:,} periods is too many to draw. Choose a coarser grouping.")
        return
    start, end = table_.period.min(), table_.period.max()
    st.caption(
        f"{dated:,} of {analyzed:,} analyzed reviews have a parseable date "
        f"({start:%Y-%m-%d} to {end:%Y-%m-%d}, UTC; dates without a time zone are read as UTC). "
        f"{periods} period(s), grouped by {dict(D='day', W='week', M='month', Y='year')[used]}. "
        "Periods with no reviews are shown as zero."
    )
    if dated < ds.MIN_TREND_ROWS or periods < 3:
        st.warning(
            "Too few dated reviews or periods for a dependable trend. Read this as a list of "
            "counts, not a direction."
        )
    st.altair_chart(charts.trend(table_, "share" if field == "Share" else "count"), width="stretch")


def rating_section(frame: pd.DataFrame) -> None:
    if "rating" not in frame:
        st.info(
            "No rating column is mapped. Ratings are optional; map one in Batch Analysis to "
            "compare stars with what the review text says."
        )
        return
    crosstab = ds.rating_table(frame)
    if crosstab.empty:
        st.info("Ratings are missing, non-numeric, or have too many distinct values to chart.")
        return
    st.altair_chart(charts.ratings(crosstab), width="stretch")
    st.caption(
        "Each bar shows the model's predicted sentiment mix for one rating value. A star rating "
        "and the text can legitimately disagree; neither is treated as ground truth here."
    )
    mismatched = ds.disagreements(frame)
    if not ds.is_star_scale(frame.rating):
        st.caption("Disagreement checks assume a 1–5 star scale, so they are skipped.")
    elif len(mismatched):
        with st.expander(f"{len(mismatched):,} reviews where stars and text disagree"):
            table(mismatched.sort_values("confidence", ascending=False).head(50))
            st.caption("Up to 50 shown, most certain predictions first.")


def number_range(series: pd.Series) -> tuple[float, float]:
    values = series.dropna()
    return float(values.min()), float(values.max())


def reset_page() -> None:
    st.session_state.ex_page = 1


def step_page(delta: int) -> None:
    st.session_state.ex_page = st.session_state.get("ex_page", 1) + delta


def explorer(predictor) -> None:
    frame, meta = active()
    if frame is None:
        empty_state(predictor)
        return
    context(meta, frame)
    with st.container(border=True):
        a, b, c = st.columns([2.2, 1.6, 1.2])
        query = a.text_input(
            "Search review text",
            key="ex_query",
            placeholder="e.g. battery, refund, flavor",
            on_change=reset_page,
        )
        labels = b.multiselect(
            "Predicted sentiment",
            [*ds.LABELS, "unavailable"],
            key="ex_labels",
            format_func=ds.badge,
            placeholder="All sentiments",
            on_change=reset_page,
        )
        min_conf = c.slider(
            "Minimum probability",
            0.0,
            1.0,
            step=0.05,
            key="ex_conf",
            on_change=reset_page,
            help="Predicted-class probability. Rows the model could not read have none.",
        )
        extras = st.columns(3)
        rating_range = products = date_range = None
        if "rating" in frame and frame.rating.notna().any():
            low, high = number_range(frame.rating)
            if low < high:
                rating_range = extras[0].slider(
                    "Rating",
                    low,
                    high,
                    key="ex_rating",
                    on_change=reset_page,
                    **initial("ex_rating", value=(low, high)),
                )
        if "product" in frame and frame["product"].notna().any():
            options = frame["product"].value_counts().head(200).index.tolist()
            products = extras[1].multiselect(
                "Product or category",
                options,
                key="ex_products",
                placeholder="All",
                on_change=reset_page,
            )
        if "date" in frame and frame.date.notna().any():
            first = frame.date.min().date()
            last = frame.date.max().date()
            picked = extras[2].date_input(
                "Date range (UTC)",
                min_value=first,
                max_value=last,
                key="ex_dates",
                on_change=reset_page,
                **initial("ex_dates", value=(first, last)),
            )
            date_range = picked if isinstance(picked, tuple) and len(picked) == 2 else None
        d, e, f, g = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
        sorts = [name for name, column in SORTS.items() if column in frame]
        sort = d.selectbox("Sort by", sorts, key="ex_sort")
        descending = e.toggle("Descending", key="ex_desc")
        size = f.selectbox(
            "Rows per page",
            PAGE_SIZES,
            key="ex_size",
            on_change=reset_page,
            **initial("ex_size", index=1),
        )
        g.button("Reset filters", on_click=reset_filters, width="stretch")

    filtered = ds.filter_reviews(
        frame,
        query=query,
        labels=labels,
        min_confidence=min_conf,
        rating_range=rating_range,
        products=products,
        date_range=date_range,
        sort_by=SORTS[sort],
        descending=descending,
    )
    pages = max(1, -(-len(filtered) // size))
    st.session_state.ex_page = min(max(1, st.session_state.get("ex_page", 1)), pages)
    rows, page, pages = ds.paginate(filtered, st.session_state.ex_page, size)
    first_row = (page - 1) * size + 1 if len(filtered) else 0
    st.markdown(
        f"**{len(filtered):,}** matching reviews of {len(frame):,}"
        + (f" · showing {first_row:,}–{first_row + len(rows) - 1:,}" if len(rows) else "")
    )
    if rows.empty:
        st.info("No reviews match these filters. Clear the search or reset filters.")
    else:
        table(rows)
    p, q, r, s = st.columns([1, 1, 1, 2], vertical_alignment="bottom")
    p.button("← Previous", on_click=step_page, args=(-1,), disabled=page <= 1, width="stretch")
    q.number_input("Page", 1, pages, key="ex_page", step=1)
    r.button("Next →", on_click=step_page, args=(1,), disabled=page >= pages, width="stretch")
    s.download_button(
        f"Download {len(filtered):,} filtered rows (CSV)",
        export_csv(filtered),
        "sentix_filtered_reviews.csv",
        "text/csv",
        disabled=filtered.empty,
        width="stretch",
    )
    if rows.empty:
        return
    options = rows.row.tolist()
    if st.session_state.get("ex_detail") not in options:
        st.session_state.pop("ex_detail", None)
    lookup = rows.set_index("row")
    chosen = st.selectbox(
        "Review details",
        options,
        index=None,
        key="ex_detail",
        placeholder="Choose a review on this page",
        format_func=lambda row: (
            f"Row {row} · {ds.badge(lookup.at[row, 'sentiment'])} · "
            f"{str(lookup.at[row, 'review_text'])[:70]}"
        ),
    )
    if chosen is not None:
        detail(lookup.loc[chosen], chosen)


def detail(row: pd.Series, number: int) -> None:
    with st.container(border=True):
        st.header(f"Review row {number}")
        label = row.sentiment
        if row.status == "ok":
            st.html(
                f'<span class="verdict {label}">{html.escape(ds.badge(label))}</span> '
                f'<span class="muted">Predicted-class probability {row.confidence:.1%}</span>'
            )
            values = {k: float(row[f"probability_{k}"]) for k in ds.LABELS}
            st.altair_chart(charts.probabilities(values), width="stretch")
            st.caption(
                "Probabilities: " + ", ".join(f"{k} {v:.1%}" for k, v in values.items()) + "."
            )
        else:
            st.warning(row.get("message") or "The model could not read this review.")
        facts = []
        for name, title in (("row_id", "ID"), ("rating", "Rating"), ("product", "Product")):
            if name in row and pd.notna(row[name]):
                facts.append(
                    f"{title}: {row[name]:g}" if name == "rating" else f"{title}: {row[name]}"
                )
        if "date" in row and pd.notna(row["date"]):
            facts.append(f"Date: {row['date']:%Y-%m-%d %H:%M} UTC")
        if row.get("route") and row.get("route") != "fast":
            facts.append(f"Read by: {row['route']}")
        if facts:
            st.text(" · ".join(facts))
        st.caption("Full review text. Use the copy button in the box's corner.")
        st.code(str(row.review_text), language=None, wrap_lines=True)


def history(predictor) -> None:
    entries = st.session_state.get("history", [])
    st.caption(
        "History lives in this browser session's server memory only. Refreshing the page or "
        f"closing the tab clears it, and nothing is written to disk. Up to {HISTORY_LIMIT} "
        "analyses are kept."
    )
    if not entries:
        st.html(
            '<div class="empty"><h2>No analyses yet</h2><p>Reviews you analyze in Review '
            "Analyzer appear here with their prediction and time.</p></div>"
        )
        st.button("Analyze a review", on_click=go, args=("Review Analyzer",), type="primary")
        return
    frame = ds.history_dataset(entries).sort_values("date", ascending=False)
    query = st.text_input("Search history", key="hist_query")
    shown = ds.filter_reviews(frame, query=query, sort_by="date", descending=True)
    st.markdown(f"**{len(shown):,}** of {len(frame):,} analyses")
    view = display_frame(shown)
    view.insert(4, "Mode", shown["mode"].to_numpy())
    view.insert(5, "Time (ms)", shown.elapsed_ms.to_numpy())
    st.dataframe(
        view,
        hide_index=True,
        column_config={
            "Probability": st.column_config.NumberColumn(format="percent"),
            "Date (UTC)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
            "Review": st.column_config.TextColumn(width="large"),
        },
    )
    a, b, c = st.columns(3)
    a.button("Use as dataset", on_click=use_history, type="primary", width="stretch")
    b.download_button(
        "Download history (CSV)",
        export_csv(frame),
        "sentix_history.csv",
        "text/csv",
        width="stretch",
    )
    with c.popover("Clear history", width="stretch"):
        st.write(f"Remove all {len(entries)} analyses from this session? This cannot be undone.")
        if st.button("Yes, clear history", type="primary"):
            st.session_state.history = []
            st.rerun()
