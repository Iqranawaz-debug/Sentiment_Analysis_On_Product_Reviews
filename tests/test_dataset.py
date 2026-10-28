"""Dashboard analytics: denominators, filters, trends and insights on known fixtures.

Fixtures are hand-built predictions, not model output; they test arithmetic only.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from sentix import dataset as ds
from sentix.bulk import analyze
from sentix.predictor import load_model


def scored(rows: list[tuple]) -> pd.DataFrame:
    """rows: (text, sentiment, confidence); 'unavailable' rows get null probability."""
    records = []
    for text, label, confidence in rows:
        ok = label != "unavailable"
        records.append(
            {
                "review_text": text,
                "sentiment": label,
                "confidence": confidence if ok else None,
                "status": "ok" if ok else "unavailable",
                "message": "" if ok else "Enter a non-empty review.",
                **{
                    f"probability_{k}": (confidence if k == label else (1 - confidence) / 2)
                    if ok
                    else None
                    for k in ds.LABELS
                },
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def frame():
    results = scored(
        [
            ("Great taste", "positive", 0.9),
            ("Awful taste", "negative", 0.8),
            ("It is fine", "neutral", 0.5),
            ("Love it", "positive", 0.95),
            ("", "unavailable", None),
        ]
    )
    source = pd.DataFrame(
        {
            "text": results.review_text,
            "stars": ["5", "1", "3", "1", "x"],
            "when": ["2026-01-01", "2026-01-02", "bad", "2026-01-20", ""],
            "sku": ["A", "B", "A", "", "B"],
        }
    )
    return ds.build_dataset(source, results, {"rating": "stars", "date": "when", "product": "sku"})


def test_build_dataset_keeps_rows_aligned(frame):
    assert frame.row.tolist() == [1, 2, 3, 4, 5]
    assert frame.rating.isna().tolist() == [False, False, False, False, True]
    assert frame.date.isna().tolist() == [False, False, True, False, True]
    assert frame["product"].isna().tolist() == [False, False, False, True, False]
    with pytest.raises(ValueError, match="one-to-one"):
        ds.build_dataset(pd.DataFrame({"a": [1]}), frame, {})


def test_kpis_use_analyzed_denominator(frame):
    stats = ds.kpis(frame, 0.7)
    assert stats["total"] == 5 and stats["analyzed"] == 4 and stats["unavailable"] == 1
    assert stats["counts"] == {"negative": 1, "neutral": 1, "positive": 2}
    assert stats["shares"]["positive"] == pytest.approx(0.5)
    assert sum(stats["shares"].values()) == pytest.approx(1)
    assert stats["mean_confidence"] == pytest.approx((0.9 + 0.8 + 0.5 + 0.95) / 4)
    assert stats["low_confidence"] == 1
    assert stats["mood_score"] == pytest.approx(50 * (1 + 0.5 - 0.25))
    assert stats["rated"] == 4 and stats["mean_rating"] == pytest.approx(2.5)


def test_empty_dataset_is_honest():
    empty = ds.build_dataset(
        pd.DataFrame({"t": ["", " "]}), scored([("", "unavailable", 0)] * 2), {}
    )
    stats = ds.kpis(empty, 0.7)
    assert stats["analyzed"] == 0
    assert stats["mean_confidence"] is None and stats["mood_score"] is None
    assert all(share is None for share in stats["shares"].values())
    assert ds.distribution(empty)["share"].tolist() == [0.0, 0.0, 0.0]
    assert ds.insights(empty, 0.7) == []
    assert ds.trend(empty)[0].empty


def test_distribution_counts_and_shares(frame):
    table = ds.distribution(frame).set_index("sentiment")
    assert table["count"].to_dict() == {"negative": 1, "neutral": 1, "positive": 2}
    assert table.share.sum() == pytest.approx(1)


def test_filters_combine(frame):
    assert ds.filter_reviews(frame, query="TASTE").row.tolist() == [1, 2]
    assert ds.filter_reviews(frame, query="taste", labels=["negative"]).row.tolist() == [2]
    assert ds.filter_reviews(frame, query="(", labels=None).empty  # literal, not regex
    assert ds.filter_reviews(frame, min_confidence=0.85).row.tolist() == [1, 4]
    assert ds.filter_reviews(frame, rating_range=(3, 5)).row.tolist() == [1, 3]
    assert ds.filter_reviews(frame, products=["A"]).row.tolist() == [1, 3]
    dated = ds.filter_reviews(frame, date_range=(date(2026, 1, 2), date(2026, 1, 20)))
    assert dated.row.tolist() == [2, 4]
    ordered = ds.filter_reviews(frame, sort_by="confidence", descending=True)
    assert ordered.row.tolist()[:2] == [4, 1] and ordered.row.iloc[-1] == 5


def test_paginate_clamps():
    rows = pd.DataFrame({"row": range(1, 53)})
    page, number, pages = ds.paginate(rows, 3, 25)
    assert (number, pages, page.row.tolist()) == (3, 3, [51, 52])
    assert ds.paginate(rows, 99, 25)[1] == 3
    assert ds.paginate(rows, 0, 25)[1] == 1
    assert ds.paginate(rows.iloc[0:0], 1, 25)[1:] == (1, 1)


def test_parse_dates_handles_iso_epoch_and_garbage():
    iso = ds.parse_dates(pd.Series(["2017-01-13T00:00:00.000Z", "2020-05-01", "nope", ""]))
    assert iso.iloc[0] == pd.Timestamp("2017-01-13", tz="UTC")
    assert iso.iloc[2:].isna().all()
    epoch = ds.parse_dates(pd.Series(["1303862400", "1346976000"]))
    assert epoch.iloc[0] == pd.Timestamp("2011-04-27", tz="UTC")
    millis = ds.parse_dates(pd.Series(["1303862400000"]))
    assert millis.iloc[0] == pd.Timestamp("2011-04-27", tz="UTC")
    assert ds.parse_dates(pd.Series(["5", "7"])).isna().all()


def test_trend_zero_fills_missing_periods(frame):
    table, freq = ds.trend(frame, "W")
    assert freq == "W"
    totals = table.drop_duplicates("period").set_index("period")["total"]
    assert totals.tolist() == [2, 0, 0, 1]  # weeks with no reviews are explicit zeros
    assert table.loc[table.total == 0, "share"].isna().all()
    assert table["count"].sum() == 3  # undated and unavailable rows excluded
    assert ds.trend(frame)[1] == "D"


def test_rating_disagreements_need_star_scale(frame):
    mismatched = ds.disagreements(frame)
    assert mismatched.row.tolist() == [4]  # 1 star, predicted positive
    assert ds.rating_table(frame).loc[5.0, "positive"] == 1
    tenpoint = frame.assign(rating=frame.rating * 2)
    assert ds.disagreements(tenpoint).empty


def test_insights_cite_sample_sizes(frame):
    found = {item["title"]: item["detail"] for item in ds.insights(frame, 0.7)}
    assert "2 of 4 analyzed reviews (50.0%)" in found["Most reviews read as positive"]
    assert "1 of 5 rows" in found["Rows the model could not read"]
    assert "1 of 4 rated reviews" in found["Stars and text disagree"]
    assert "Negative share over time" not in found  # too few dated rows to compare


def test_frequent_terms_keep_negation_and_count_documents():
    terms = dict(ds.frequent_terms(["not good not good", "not working", "the box"]))
    assert terms["not"] == 2 and terms["good"] == 1
    assert "the" not in terms


def test_history_round_trip():
    result = {
        "label": "positive",
        "confidence": 0.9,
        "status": "ok",
        "probabilities": {"negative": 0.05, "neutral": 0.05, "positive": 0.9},
    }
    frame = ds.history_dataset([ds.history_entry("Nice", result, "Fast Mode", 3.21)] * 2)
    assert frame.row.tolist() == [1, 2]
    assert str(frame.date.dt.tz) == "UTC"
    assert ds.kpis(frame, 0.7)["counts"]["positive"] == 2


def test_dataset_export_is_formula_safe_and_row_aligned(frame):
    from src.predict import export_csv

    risky = frame.assign(review_text=["=HYPERLINK(1)", *frame.review_text[1:]])
    exported = pd.read_csv(pd.io.common.BytesIO(export_csv(risky)), encoding="utf-8-sig")
    assert exported.review_text.iloc[0] == "'=HYPERLINK(1)"
    assert exported.row.tolist() == [1, 2, 3, 4, 5]
    assert exported.message.iloc[4] == "Enter a non-empty review."


def test_guess_column():
    names = ["Id", "ProductId", "Score", "Time", "Text"]
    assert ds.guess_column(names, "rating") == "Score"
    assert ds.guess_column(names, "date") == "Time"
    assert ds.guess_column(names, "product") == "ProductId"
    assert ds.guess_column(names, "row_id") == "Id"
    assert ds.guess_column(["a"], "rating") is None


def test_badge_never_relies_on_color():
    assert ds.badge("positive") == "↑ Positive"
    assert ds.badge("unavailable") == "∅ Unavailable"


def test_sample_dataset_uses_real_model_and_ratings():
    source, mapping = ds.load_sample()
    assert len(source) == 2698 and mapping == {"rating": "star_rating", "product": "source"}
    head = source.head(40).reset_index(drop=True)
    frame = ds.build_dataset(head, analyze(load_model(), head.review_text.tolist()), mapping)
    assert set(frame.sentiment) <= {*ds.LABELS, "unavailable"}
    ok = frame[frame.status == "ok"]
    probabilities = ok[[f"probability_{k}" for k in ds.LABELS]].sum(axis=1)
    assert np.allclose(probabilities, 1)
    assert frame.rating.between(1, 5).all()


def test_display_text_only_converts_line_break_tags():
    assert ds.display_text("Good.<br /><br />Fast<BR>ok") == "Good.\n\nFast\nok"
    assert ds.display_text("<script>x</script>") == "<script>x</script>"  # still plain text
