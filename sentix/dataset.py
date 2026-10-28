"""Active-dataset analytics over real model output. Pure functions; no invented values.

Shares always use analyzed (status == "ok") rows as the denominator. Ratings, dates and
products come only from columns the user mapped; nothing is inferred when they are absent.
"""

import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.preprocessing import clean_text

LABELS = ("negative", "neutral", "positive")
COLORS = {"positive": "#1E7F4F", "neutral": "#B7791F", "negative": "#C0392B"}
ICONS = {"positive": "↑", "neutral": "→", "negative": "↓", "unavailable": "∅"}
OPTIONAL = ("rating", "date", "product", "row_id")
GUESSES = {
    "rating": ("rating", "star_rating", "stars", "score", "reviews.rating", "overall"),
    "date": ("date", "time", "timestamp", "reviews.date", "review_date", "created_at"),
    "product": ("product", "productid", "product_id", "asin", "asins", "name", "category"),
    "row_id": ("id", "review_id", "reviewid"),
}
# Negations carry sentiment, so they stay visible in frequent-term summaries.
STOP_WORDS = ENGLISH_STOP_WORDS - {"not", "no", "nor", "never", "nothing", "very"}
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "reports/approved_20261016/test.csv"
SAMPLE_NAME = "Sample: 2,698 real Amazon reviews"
MIN_TREND_ROWS = 30


def display_text(text) -> str:
    """Readable review text for display only: source reviews contain literal <br /> tags.

    The model always receives the original text; this never feeds predictions.
    """
    return re.sub(r"(?i)<br\s*/?>", "\n", str(text))


def badge(label: str) -> str:
    """Text plus a direction glyph, so meaning never depends on color alone."""
    return f"{ICONS.get(label, '∅')} {label.title()}"


def mood_score(frame: pd.DataFrame) -> float | None:
    valid = frame[frame.status == "ok"]
    if valid.empty:
        return None
    return 50 * (
        1 + (valid.sentiment == "positive").mean() - (valid.sentiment == "negative").mean()
    )


def guess_column(names: list[str], kind: str) -> str | None:
    lowered = {name.lower(): name for name in names}
    return next((lowered[g] for g in GUESSES[kind] if g in lowered), None)


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse ISO/mixed strings or Unix epochs to UTC; unparseable values become NaT."""
    text = pd.Series(values, dtype="string").str.strip().replace("", pd.NA)
    present = text.notna()
    numeric = pd.to_numeric(text, errors="coerce")
    if present.any() and numeric[present].notna().all():
        for unit, low, high in (("s", 1e8, 4e9), ("ms", 1e11, 4e12)):
            if numeric[present].between(low, high).all():
                return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
        return pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(text, errors="coerce", utc=True, format="mixed")


def build_dataset(source: pd.DataFrame, results: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Attach mapped metadata to predictions row-for-row; `row` is the 1-based data row."""
    if len(source) != len(results):
        raise ValueError("Predictions must align one-to-one with input rows.")
    frame = results.reset_index(drop=True).copy()
    source = source.reset_index(drop=True)
    frame.insert(0, "row", np.arange(1, len(frame) + 1))
    if mapping.get("row_id"):
        frame.insert(1, "row_id", source[mapping["row_id"]].astype(str))
    if mapping.get("rating"):
        frame["rating"] = pd.to_numeric(source[mapping["rating"]], errors="coerce")
    if mapping.get("date"):
        frame["date"] = parse_dates(source[mapping["date"]])
    if mapping.get("product"):
        frame["product"] = source[mapping["product"]].astype(str).str.strip().replace("", pd.NA)
    if "message" not in frame:
        frame["message"] = ""
    return frame


def history_entry(text: str, result: dict, mode: str, elapsed_ms: float, model: str = "") -> dict:
    """A frozen record: later model changes never rewrite an old prediction."""
    return {
        "review_text": text,
        "sentiment": result["label"],
        "confidence": result["confidence"],
        "status": result["status"],
        "route": result.get("route", "fast"),
        "warning": result.get("warning", ""),
        "message": result.get("message", ""),
        **{f"probability_{k}": v for k, v in result["probabilities"].items()},
        "mode": mode,
        "model": model,
        "uncertainty": ", ".join(result.get("uncertainty", [])),
        "feedback": "",
        "elapsed_ms": round(elapsed_ms, 1),
        "date": datetime.now(timezone.utc),
    }


def history_dataset(entries: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(entries)
    frame.insert(0, "row", np.arange(1, len(frame) + 1))
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame


def load_sample(path: Path = SAMPLE_PATH) -> tuple[pd.DataFrame, dict]:
    """Real tracked reviews with star ratings and a source category; no dates exist."""
    frame = pd.read_csv(path, usecols=["review_text", "star_rating", "source"], dtype=str)
    return frame.fillna(""), {"rating": "star_rating", "product": "source"}


def kpis(frame: pd.DataFrame, low_threshold: float) -> dict:
    ok = frame[frame.status == "ok"]
    analyzed = len(ok)
    counts = ok.sentiment.value_counts().reindex(LABELS, fill_value=0)
    result = {
        "total": len(frame),
        "analyzed": analyzed,
        "unavailable": len(frame) - analyzed,
        "counts": {k: int(v) for k, v in counts.items()},
        "shares": {k: (v / analyzed if analyzed else None) for k, v in counts.items()},
        "mean_confidence": float(ok.confidence.mean()) if analyzed else None,
        "low_confidence": int((ok.confidence < low_threshold).sum()),
        "mood_score": mood_score(frame),
        "rated": 0,
        "mean_rating": None,
    }
    if "rating" in frame:
        rated = frame.rating.dropna()
        result["rated"] = len(rated)
        result["mean_rating"] = float(rated.mean()) if len(rated) else None
    return result


def distribution(frame: pd.DataFrame) -> pd.DataFrame:
    ok = frame[frame.status == "ok"]
    counts = ok.sentiment.value_counts().reindex(LABELS, fill_value=0)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "sentiment": list(LABELS),
            "count": counts.astype(int).tolist(),
            "share": [(c / total if total else 0.0) for c in counts],
        }
    )


def filter_reviews(
    frame: pd.DataFrame,
    query: str = "",
    labels: list[str] | None = None,
    min_confidence: float = 0.0,
    rating_range: tuple[float, float] | None = None,
    products: list[str] | None = None,
    date_range: tuple | None = None,
    sort_by: str = "row",
    descending: bool = False,
) -> pd.DataFrame:
    """All filters combine (AND). Search is a literal, case-insensitive substring."""
    mask = pd.Series(True, index=frame.index)
    if query.strip():
        mask &= frame.review_text.astype(str).str.contains(
            query.strip(), case=False, regex=False, na=False
        )
    if labels:
        mask &= frame.sentiment.isin(labels)
    if min_confidence > 0:
        mask &= frame.confidence.fillna(-1) >= min_confidence
    if rating_range and "rating" in frame:
        mask &= frame.rating.between(*rating_range)
    if products and "product" in frame:
        mask &= frame["product"].isin(products)
    if date_range and "date" in frame:
        start = pd.Timestamp(date_range[0], tz="UTC")
        end = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
        mask &= (frame.date >= start) & (frame.date < end)
    result = frame[mask]
    if sort_by in result:
        result = result.sort_values(sort_by, ascending=not descending, na_position="last")
    return result


def paginate(frame: pd.DataFrame, page: int, size: int) -> tuple[pd.DataFrame, int, int]:
    """Clamp the page; return (rows, page, page_count). An empty frame has one page."""
    pages = max(1, math.ceil(len(frame) / size))
    page = min(max(1, page), pages)
    return frame.iloc[(page - 1) * size : page * size], page, pages


def trend_frequency(dates: pd.Series) -> str:
    span = (dates.max() - dates.min()).days if len(dates) else 0
    return "D" if span <= 45 else "W" if span <= 180 else "M" if span <= 3650 else "Y"


def trend(frame: pd.DataFrame, freq: str | None = None) -> tuple[pd.DataFrame, str]:
    """Counts per period and class; empty periods inside the range are real zeros."""
    if "date" not in frame:
        return pd.DataFrame(columns=["period", "sentiment", "count", "share", "total"]), ""
    dated = frame[(frame.status == "ok") & frame.date.notna()]
    if dated.empty:
        return pd.DataFrame(columns=["period", "sentiment", "count", "share", "total"]), ""
    freq = freq or trend_frequency(dated.date)
    periods = dated.date.dt.tz_convert(None).dt.to_period(freq)
    table = pd.crosstab(periods, dated.sentiment).reindex(columns=list(LABELS), fill_value=0)
    full = pd.period_range(periods.min(), periods.max(), freq=freq)
    table = table.reindex(full, fill_value=0)
    totals = table.sum(axis=1)
    long = table.reset_index(names="period").melt(
        id_vars="period", var_name="sentiment", value_name="count"
    )
    long["total"] = long.period.map(totals).astype(int)
    long["share"] = np.where(long.total > 0, long["count"] / long.total.replace(0, 1), np.nan)
    long["period"] = long.period.dt.start_time
    return long, freq


def is_star_scale(ratings: pd.Series) -> bool:
    ratings = ratings.dropna()
    return bool(len(ratings)) and ratings.between(1, 5).all()


def rating_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Predicted sentiment counts per observed rating value (analyzed, rated rows)."""
    if "rating" not in frame:
        return pd.DataFrame()
    rows = frame[(frame.status == "ok") & frame.rating.notna()]
    if rows.empty or rows.rating.nunique() > 11:
        return pd.DataFrame()
    return pd.crosstab(rows.rating, rows.sentiment).reindex(columns=list(LABELS), fill_value=0)


LENGTH_BINS = [0, 8, 30, 80, 200, np.inf]
LENGTH_NAMES = ["1–8 words", "9–30 words", "31–80 words", "81–200 words", "200+ words"]


def segment_table(frame: pd.DataFrame, by: str, top: int = 8) -> pd.DataFrame:
    """Predicted-sentiment counts per product/category (top N) or per review-length band."""
    ok = frame[frame.status == "ok"]
    if by == "product":
        ok = ok[ok["product"].notna()]
        keep = ok["product"].value_counts().head(top).index
        segments = ok["product"].where(ok["product"].isin(keep))
        ok, segments = ok[segments.notna()], segments.dropna()
        order = list(keep)
    else:
        words = ok.review_text.astype(str).str.split().str.len()
        segments = pd.cut(words, LENGTH_BINS, labels=LENGTH_NAMES).astype(str)
        order = [name for name in LENGTH_NAMES if name in set(segments)]
    if ok.empty:
        return pd.DataFrame(columns=["segment", "sentiment", "count", "share", "total"])
    table = pd.crosstab(segments, ok.sentiment).reindex(columns=list(LABELS), fill_value=0)
    table = table.reindex(order)
    long = table.reset_index(names="segment").melt(
        id_vars="segment", var_name="sentiment", value_name="count"
    )
    long["total"] = long.segment.map(table.sum(axis=1)).astype(int)
    long["share"] = long["count"] / long.total.where(long.total > 0, 1)
    return long


def disagreements(frame: pd.DataFrame) -> pd.DataFrame:
    """High stars with negative text, or low stars with positive text (1–5 scale only)."""
    if "rating" not in frame or not is_star_scale(frame.rating):
        return frame.iloc[0:0]
    ok = frame[frame.status == "ok"]
    return ok[
        ((ok.rating >= 4) & (ok.sentiment == "negative"))
        | ((ok.rating <= 2) & (ok.sentiment == "positive"))
    ]


def frequent_terms(texts, top: int = 8) -> list[tuple[str, int]]:
    """Document frequency of normalized words: descriptive, not a significance test."""
    counts: dict[str, int] = {}
    for text in texts:
        for word in set(clean_text(str(text)).split()):
            if len(word) > 2 and word not in STOP_WORDS and not word.isdigit():
                counts[word] = counts.get(word, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top]


def insights(frame: pd.DataFrame, low_threshold: float) -> list[dict]:
    """Each insight states its sample size; nothing is emitted without supporting rows."""
    found = []
    stats = kpis(frame, low_threshold)
    n = stats["analyzed"]
    if not n:
        return found
    top = max(LABELS, key=lambda label: stats["counts"][label])
    found.append(
        {
            "title": f"Most reviews read as {top}",
            "detail": f"{stats['counts'][top]:,} of {n:,} analyzed reviews "
            f"({stats['shares'][top]:.1%}) were predicted {top}.",
        }
    )
    found.append(
        {
            "title": "Negative share",
            "detail": f"{stats['counts']['negative']:,} of {n:,} analyzed reviews "
            f"({stats['shares']['negative']:.1%}) were predicted negative.",
        }
    )
    if stats["low_confidence"]:
        found.append(
            {
                "title": "Predictions to double-check",
                "detail": f"{stats['low_confidence']:,} of {n:,} predictions have a predicted-"
                f"class probability below {low_threshold:.0%}; treat those labels as tentative.",
            }
        )
    if stats["unavailable"]:
        found.append(
            {
                "title": "Rows the model could not read",
                "detail": f"{stats['unavailable']:,} of {stats['total']:,} rows were empty, too "
                "long or had no familiar words. They are excluded from every share.",
            }
        )
    normalized = frame.review_text.astype(str).map(clean_text)
    duplicates = int(normalized[normalized != ""].duplicated().sum())
    if duplicates:
        found.append(
            {
                "title": "Repeated review text",
                "detail": f"{duplicates:,} rows repeat an earlier review after normalization. "
                "They are kept and counted; check for double imports.",
            }
        )
    mismatched = disagreements(frame)
    if len(mismatched):
        rated = int(frame.loc[frame.status == "ok", "rating"].notna().sum())
        found.append(
            {
                "title": "Stars and text disagree",
                "detail": f"{len(mismatched):,} of {rated:,} rated reviews pair 4–5 stars with "
                "negative text or 1–2 stars with positive text. Either may be right; "
                "these are worth reading.",
            }
        )
    periods, _ = trend(frame)
    if not periods.empty:
        totals = periods.drop_duplicates("period").set_index("period")["total"]
        half = len(totals) // 2
        if half:
            early = periods[periods.period.isin(totals.index[:half])]
            late = periods[periods.period.isin(totals.index[half:])]
            early_n, late_n = int(totals.iloc[:half].sum()), int(totals.iloc[half:].sum())
            if early_n >= MIN_TREND_ROWS and late_n >= MIN_TREND_ROWS:
                before = early.loc[early.sentiment == "negative", "count"].sum() / early_n
                after = late.loc[late.sentiment == "negative", "count"].sum() / late_n
                found.append(
                    {
                        "title": "Negative share over time",
                        "detail": f"Earlier half of the period: {before:.1%} negative "
                        f"(n={early_n:,}); later half: {after:.1%} (n={late_n:,}). "
                        "Descriptive only; no significance test.",
                    }
                )
    negative = frame[(frame.status == "ok") & (frame.sentiment == "negative")]
    if len(negative) >= 10:
        terms = frequent_terms(negative.review_text)
        found.append(
            {
                "title": "Frequent words in negative reviews",
                "detail": ", ".join(f"{w} ({c})" for w, c in terms)
                + f" — reviews containing each word, out of {len(negative):,} negative reviews.",
            }
        )
    return found
