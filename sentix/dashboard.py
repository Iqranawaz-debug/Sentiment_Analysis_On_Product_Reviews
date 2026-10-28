"""Overview, Review Explorer and History over the active dataset.

The active dataset is either a scored CSV (Batch Analysis), the bundled sample, or this
session's analyzer history. It lives in Streamlit session memory only.
"""

import html
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from sentix import charts
from sentix import dataset as ds
from sentix.bulk import analyze
from sentix.mascot import senti_img, svg_image, unsure_threshold
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
    "Real Amazon reviews and star ratings from a tracked held-out split "
    "(reports/approved_20261016/test.csv) that excludes every review in the production "
    "model's 30,000-review sample and near-duplicates of them. Scored live by the Fast "
    "Mode model. Ratings here are not used as accuracy: see Model & Evaluation for the "
    "measured results. It has no review dates."
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
    st.toast(f"Sample loaded: {len(frame):,} reviews scored.", icon=":material/check_circle:")


def empty(title: str, body: str) -> None:
    """Shared empty state with a small, static Senti welcoming the user."""
    st.html(
        f'<div class="empty">{senti_img("welcome")}<div><h2>{html.escape(title)}</h2>'
        f"<p>{html.escape(body)}</p></div></div>"
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
        st.toast(
            f"Using {len(entries)} analyzed reviews as the dataset.", icon=":material/check_circle:"
        )


def record_history(text: str, result: dict, mode: str, elapsed_ms: float, model: str = "") -> None:
    entries = st.session_state.setdefault("history", [])
    entries.append(ds.history_entry(text, result, mode, elapsed_ms, model))
    del entries[:-HISTORY_LIMIT]


def delete_history(number: int) -> None:
    """`number` is the 1-based position shown in the table's Row column."""
    entries = st.session_state.get("history", [])
    if 1 <= number <= len(entries):
        del entries[number - 1]
    st.session_state.pop("hist_detail", None)


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
        frame.review_text.map(lambda t: " ".join(ds.display_text(t).split()))
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
    empty(
        "No dataset loaded yet",
        "The overview summarizes a set of reviews scored by the model. Upload a CSV in Batch "
        "Analysis, load the bundled sample of real reviews, or use the reviews you analyzed "
        "in this session.",
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


WELCOME_TILES = (
    ("Sentiment mix", "How many reviews read as positive, neutral or negative."),
    ("Insights", "Short notes on what stands out, each with the sample size behind it."),
    ("Ratings and trends", "Stars versus sentiment, plus trends when your data has dates."),
)


def overview_welcome(predictor) -> None:
    """Overview-only empty state: a welcoming card with the two ways to start.

    Review Explorer keeps `empty_state`; the actions reuse the same handlers.
    """
    rows = max(1, int(os.environ.get("SENTIX_MAX_ROWS", "5000")))
    with st.container(border=True, key="ov_welcome"):
        art, body = st.columns([1, 2.6], vertical_alignment="center", gap="large")
        art.html(f'<div class="welcome-art">{senti_img("welcome", 120)}</div>')
        with body:
            st.html(
                '<span class="eyebrow">Get started</span>'
                '<h2 class="welcome-title">Let\'s look at some reviews</h2>'
                '<p class="welcome-copy">The overview summarizes a set of reviews scored by '
                "the sentiment model: the overall mood, what stands out, and how sure the model "
                "is. Pick a dataset to begin.</p>"
            )
            sample, upload = st.columns(2)
            with sample:
                if st.button(
                    "Load sample dataset",
                    key="ov_load_sample",
                    type="primary",
                    icon=":material/dataset:",
                    width="stretch",
                ):
                    load_sample_dataset(predictor)
                    st.rerun()
                st.caption("2,698 real Amazon reviews with star ratings, scored when you click.")
            with upload:
                st.button(
                    "Upload a CSV",
                    key="ov_upload",
                    icon=":material/upload_file:",
                    on_click=go,
                    args=("Batch Analysis",),
                    width="stretch",
                )
                st.caption(f"Use your own reviews, up to {rows:,} rows. Opens Batch Analysis.")
        st.html(
            '<div class="welcome-tiles" role="list" aria-label="What you will see">'
            + "".join(
                f'<div class="welcome-tile" role="listitem"><strong>{html.escape(title)}</strong>'
                f"<span>{html.escape(text)}</span></div>"
                for title, text in WELCOME_TILES
            )
            + "</div>"
        )
        entries = len(st.session_state.get("history", []))
        if entries:
            st.button(
                f"Or use the {entries} review{'s' if entries != 1 else ''} you analyzed "
                "this session",
                key="ov_use_history",
                type="tertiary",
                icon=":material/history:",
                on_click=use_history,
            )


FACES = {
    # Small faces in the sentiment colours; the label text beside them carries the meaning.
    "positive": ("#1E7A4C", "M10 18q6 6 12 0"),
    "neutral": ("#8A5A0B", "M10 19h12"),
    "negative": ("#B5382B", "M10 21q6-6 12 0"),
}


def face_icon(label: str) -> str:
    colour, mouth = FACES[label]
    return svg_image(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none" '
        f'stroke="{colour}" stroke-width="2.4" stroke-linecap="round">'
        f'<circle cx="16" cy="16" r="13"/><circle cx="11.5" cy="13" r="1.4" fill="{colour}"/>'
        f'<circle cx="20.5" cy="13" r="1.4" fill="{colour}"/><path d="{mouth}"/></svg>',
        "",
        22,
    )


def distribution_section(summary: dict) -> None:
    """Live counts and shares of predicted sentiment for the active dataset.

    Recomputed from the active dataset on every run, so it changes with the data.
    """
    n = summary["eligible"]
    excluded = [f"{summary['unreadable']:,} unreadable rows excluded"]
    if summary["invalid"]:
        excluded.append(f"{summary['invalid']:,} rows with an unexpected label excluded")
    with st.container(border=True, key="ov_dist"):
        st.header("Sentiment distribution")
        st.html(
            '<p class="dist-sub"><span class="live" title="Recalculated from the active '
            'dataset">Live</span>'
            f"Based on {n:,} analyzed review{'s' if n != 1 else ''} · "
            f"{html.escape(' · '.join(excluded))}</p>"
        )
        if not n:
            st.info(
                "No review in this dataset could be analyzed, so there is no distribution to show."
            )
            return
        st.html(
            '<div class="dist-grid">'
            + "".join(
                f'<div class="dist-card {label}"><div class="dist-top">{face_icon(label)}'
                f'<span class="dist-label">{html.escape(ds.badge(label))}</span></div>'
                f'<div class="dist-count">{summary["counts"][label]:,}</div>'
                f'<div class="dist-pct"><b>{summary["percent"][label]:.1f}%</b> of '
                f"{n:,} eligible reviews</div></div>"
                for label in charts.PROPORTION_ORDER
            )
            + "</div>"
        )
        st.altair_chart(charts.proportion(summary), width="stretch")
        st.caption(
            "Predicted sentiment of eligible reviews: "
            + "; ".join(
                f"{label} {summary['counts'][label]:,} ({summary['percent'][label]:.1f}%)"
                for label in charts.PROPORTION_ORDER
            )
            + ". Labels are the model's predictions, not star ratings."
            + (
                " Percentages are rounded to one decimal, so they may not add up to exactly 100%."
                if summary["rounding_gap"]
                else ""
            )
        )


def overview(predictor) -> None:
    frame, meta = active()
    if frame is None:
        overview_welcome(predictor)
        return
    context(meta, frame)
    low = unsure_threshold()
    stats = ds.kpis(frame, low)
    n = stats["analyzed"]
    distribution_section(ds.sentiment_summary(frame))
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

    with st.container(border=True):
        st.header("Insights")
        insight_list(ds.insights(frame, low))

    with st.container(border=True):
        if "date" in frame:
            st.header("Sentiment over time")
            trend_section(frame)
        else:
            segment_section(frame)

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

    status_section(frame, meta, predictor)


def segment_section(frame: pd.DataFrame) -> None:
    """Useful breakdown when no dates exist: by category if mapped, else by length."""
    by_product = "product" in frame and frame["product"].notna().any()
    title = "Product or category" if by_product else "Review length"
    st.header(f"Sentiment by {title.lower()}")
    st.info(
        "Trends need real review dates, and this dataset has none mapped, so dates are never "
        f"invented. Showing sentiment by {title.lower()} instead."
    )
    table_ = ds.segment_table(frame, "product" if by_product else "length")
    if table_.empty:
        st.caption("No analyzed reviews to break down.")
        return
    st.altair_chart(charts.segments(table_, title), width="stretch")
    biggest = table_.drop_duplicates("segment")
    st.caption(
        "Predicted sentiment mix within each "
        + ("category (largest 8 shown)" if by_product else "word-count band")
        + ". Segments: "
        + "; ".join(f"{s} n={n:,}" for s, n in zip(biggest.segment, biggest.total))
        + ". Small segments can swing widely."
    )


def evaluation_status() -> list[str]:
    """Which saved evaluations exist; this is not a live health check."""
    reports = Path(
        os.environ.get("SENTIMENT_REPORTS_DIR", Path(__file__).resolve().parents[1] / "reports")
    )
    found = []
    try:
        policy = json.loads((reports / "decision_policy.json").read_text(encoding="utf-8"))
        found.append(f"Decision rule evaluated {policy['created_utc'][:10]} on held-out data")
    except OSError, ValueError, KeyError:
        found.append("No decision-rule evaluation saved")
    deep = reports / "deep_mode_e2e_20261028_batch1/results.json"
    found.append(
        "Deep Mode measured end to end 2026-10-28" if deep.exists() else "Deep Mode not measured"
    )
    return found


def status_section(frame: pd.DataFrame, meta: dict, predictor) -> None:
    stats = ds.kpis(frame, unsure_threshold())
    normalized = frame.review_text.astype(str).str.lower().str.strip()
    duplicates = int(normalized[normalized != ""].duplicated().sum())
    warnings = []
    if stats["unavailable"]:
        warnings.append(f"{stats['unavailable']:,} rows could not be analyzed")
    if duplicates:
        warnings.append(f"{duplicates:,} repeated review texts")
    for column, name in (("rating", "ratings"), ("date", "dates")):
        if column in frame and frame[column].isna().any():
            warnings.append(f"{int(frame[column].isna().sum()):,} missing or unreadable {name}")
    facts = {
        "Active dataset": meta["name"],
        "Records": f"{stats['total']:,} rows; {stats['analyzed']:,} analyzed",
        "Scored with": meta["mode"],
        "Fast model": f"TF-IDF logistic regression, artifact "
        f"{predictor.metadata['artifact_sha256'][:12]}; labels by {predictor.decision_rule}",
        "Saved evaluations": "; ".join(evaluation_status()),
        "Data warnings": "; ".join(warnings) or "None found",
    }
    with st.container(border=True):
        st.header("Data and model status")
        st.html(
            '<dl class="facts">'
            + "".join(
                f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in facts.items()
            )
            + "</dl>"
        )
        st.caption(
            "Saved evaluations describe measured quality; they are not a live availability "
            "check. Theme extraction is not implemented, so no praise or complaint themes are "
            "shown; frequent words under Insights are descriptive only."
        )


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
        for name, title in (("mode", "Mode"), ("model", "Model"), ("feedback", "Feedback")):
            if name in row and isinstance(row[name], str) and row[name]:
                facts.append(f"{title}: {row[name]}")
        if facts:
            st.text(" · ".join(facts))
        st.caption("Full review text. Use the copy button in the box's corner.")
        st.code(ds.display_text(row.review_text), language=None, wrap_lines=True)


def history(predictor) -> None:
    entries = st.session_state.get("history", [])
    st.caption(
        "History lives in this browser session's server memory only. Refreshing the page or "
        f"closing the tab clears it, and nothing is written to disk. Up to {HISTORY_LIMIT} "
        "analyses are kept."
    )
    if not entries:
        empty(
            "No analyses yet",
            "Reviews you analyze in Review Analyzer appear here with their prediction, model "
            "and time.",
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
    for column, title in (("model", "Model"), ("feedback", "Feedback")):
        if column in shown:
            view[title] = shown[column].fillna("").to_numpy()
    st.dataframe(
        view,
        hide_index=True,
        column_config={
            "Probability": st.column_config.NumberColumn(format="percent"),
            "Date (UTC)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
            "Review": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "Each analysis keeps the model and mode that produced it; later model changes do not "
        "rewrite old results."
    )
    options = shown.row.tolist()
    if st.session_state.get("hist_detail") not in options:
        st.session_state.pop("hist_detail", None)
    lookup = frame.set_index("row")
    chosen = st.selectbox(
        "Open an analysis",
        options,
        index=None,
        key="hist_detail",
        placeholder="Choose an analysis to see its details",
        format_func=lambda row: (
            f"Row {row} · {ds.badge(lookup.at[row, 'sentiment'])} · "
            f"{lookup.at[row, 'date']:%H:%M:%S} · {str(lookup.at[row, 'review_text'])[:60]}"
        ),
    )
    if chosen is not None:
        detail(lookup.loc[chosen], chosen)
        st.button(
            "Delete this analysis",
            icon=":material/delete:",
            on_click=delete_history,
            args=(chosen,),
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
