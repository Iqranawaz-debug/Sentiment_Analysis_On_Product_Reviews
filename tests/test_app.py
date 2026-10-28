from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from sentix.state import slug
from sentix.views import EXAMPLES, NAV, PAGES

ROOT = Path(__file__).resolve().parents[1]
FALLBACK = "couldn't open this feature"
PRAISE, COMPLAINT, LUKEWARM = EXAMPLES


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(ROOT / "models"))
    monkeypatch.setenv("SENTIMENT_REPORTS_DIR", str(ROOT / "reports"))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run(timeout=60)


def button(app, label):
    return next(b for b in app.button if b.label == label)


def visit(app, page):
    app.button(key=f"nav_{slug(page)}").click().run(timeout=60)
    assert not app.exception
    assert app.session_state.page == page
    assert not any(FALLBACK in info.value for info in app.info), page
    return app


def analyze(app, example):
    button(app, example).click().run()
    button(app, "Analyze review").click().run()
    assert not app.exception
    return app.session_state.result


def html_text(app):
    return " ".join(e.proto.body for e in app.get("html"))


def test_missing_model_screen(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run(timeout=30)
    assert not app.exception
    assert "could not be loaded" in app.error[0].value


def test_every_page_renders_and_nav_marks_the_active_page(app):
    assert app.session_state.page == "Review Analyzer"
    assert [p for group in NAV.values() for p in group] == list(PAGES)
    for page in PAGES:
        visit(app, page)
        for other in PAGES:
            expected = "primary" if other == page else "tertiary"
            assert app.button(key=f"nav_{slug(other)}").proto.type == expected
        assert app.query_params["page"] == [slug(page)]
    visit(app, "Overview")
    assert app.button(key="ov_load_sample").proto.type == "primary"
    assert app.button(key="ov_upload")


def test_overview_welcome_card(app):
    visit(app, "Overview")
    page = html_text(app)
    assert "Let's look at some reviews" in page
    assert page.count('class="welcome-tile"') == 3
    assert 'alt=""' in page  # the mascot is decorative
    # No session analyses yet: the optional third route is not shown at all.
    assert not [b for b in app.button if b.key == "ov_use_history"]
    assert any("up to 5,000 rows" in c.value for c in app.caption)


def test_overview_welcome_upload_opens_batch_analysis(app):
    visit(app, "Overview")
    app.button(key="ov_upload").click().run()
    assert not app.exception
    assert app.session_state.page == "Batch Analysis"
    assert app.query_params["page"] == ["batch-analysis"]


def test_overview_welcome_loads_sample(app):
    visit(app, "Overview")
    app.button(key="ov_load_sample").click().run(timeout=120)
    assert not app.exception
    assert app.session_state.dataset_meta["kind"] == "sample"
    assert len(app.session_state.dataset) == 2698
    assert "Sentiment distribution" in [h.value for h in app.header]
    assert not [b for b in app.button if b.key == "ov_load_sample"]


def distribution_html(app):
    return " ".join(
        e.proto.body for e in app.get("html") if "dist-" in e.proto.body or "live" in e.proto.body
    )


def test_distribution_matches_active_dataset_and_updates(app):
    analyze(app, COMPLAINT)
    visit(app, "Overview")
    app.button(key="ov_load_sample").click().run(timeout=120)
    frame = app.session_state.dataset
    ok = frame[frame.status == "ok"]
    page = distribution_html(app)
    assert f"Based on {len(ok):,} analyzed reviews" in page
    assert f"{(frame.status != 'ok').sum():,} unreadable rows excluded" in page
    for label in ("positive", "neutral", "negative"):
        count = int((ok.sentiment == label).sum())
        assert f'<div class="dist-count">{count:,}</div>' in page
        assert f"<b>{count / len(ok) * 100:.1f}%</b> of {len(ok):,} eligible reviews" in page
    # The section follows the active dataset: switch to this session's one analysis.
    visit(app, "History")
    button(app, "Use as dataset").click().run()
    visit(app, "Overview")
    page = distribution_html(app)
    assert "Based on 1 analyzed review " in page
    assert '<div class="dist-count">1</div>' in page and "<b>100.0%</b>" in page


def test_distribution_with_no_readable_reviews(app):
    visit(app, "Batch Analysis")
    app.file_uploader[0].set_value(("blank.csv", b"review_text\n!!!\n12345\n", "text/csv")).run()
    button(app, "Analyze CSV").click().run()
    visit(app, "Overview")
    assert "Based on 0 analyzed reviews" in distribution_html(app)
    assert any("no distribution to show" in i.value for i in app.info)
    assert 'class="dist-card' not in distribution_html(app)


def test_overview_welcome_offers_session_reviews_once_they_exist(app):
    analyze(app, PRAISE)
    visit(app, "Overview")
    link = app.button(key="ov_use_history")
    assert link.label == "Or use the 1 review you analyzed this session"
    link.click().run()
    assert not app.exception
    assert app.session_state.dataset_meta["kind"] == "history"


def test_bookmarked_page_link_opens_that_page(monkeypatch):
    monkeypatch.setenv("SENTIMENT_REPORTS_DIR", str(ROOT / "reports"))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.query_params["page"] = "model-evaluation"
    app.run(timeout=60)
    assert not app.exception
    assert app.session_state.page == "Model & Evaluation"


def test_examples_only_load_text(app):
    button(app, PRAISE).click().run()
    assert app.text_area(key="review").value == EXAMPLES[PRAISE]
    assert app.session_state.result is None
    assert not app.session_state.get("history")


def test_app_pages_and_predictions(app):
    assert not app.exception
    assert not app.error
    for example, expected in [(PRAISE, "positive"), (COMPLAINT, "negative"), (LUKEWARM, "neutral")]:
        assert analyze(app, example)["label"] == expected
    history = app.session_state.history
    assert [entry["sentiment"] for entry in history] == ["positive", "negative", "neutral"]
    assert all("TF-IDF logistic regression" in entry["model"] for entry in history)
    assert app.session_state.result_meta["elapsed_ms"] >= 0
    visit(app, "Explain")
    button(app, "Explain this review").click().run()
    assert not app.exception
    assert app.session_state.explanation[0] == app.session_state.review
    visit(app, "Guess the Mood")
    button(app, "Positive").click().run()
    assert not app.exception
    assert app.session_state.game_reveal is not None
    assert app.session_state.result["label"] == "neutral"
    button(app, "Reset score").click().run()
    assert app.session_state.game_score == [0, 0]
    visit(app, "Review Analyzer")
    assert app.text_area(key="review").value == app.session_state.analyzed_text


def test_result_reading_matches_the_tentative_rule(app):
    analyze(app, PRAISE)
    page = html_text(app)
    assert "reads this review as <b>positive</b>" in page and "result-foot" in page
    app.text_area(key="review").input("Absolutely hate it").run()
    button(app, "Analyze review").click().run()
    page = html_text(app)
    assert "leans <b>positive</b>, but it is not sure" in page


def test_editing_clears_the_old_result(app):
    analyze(app, PRAISE)
    app.text_area(key="review").input("Terrible, it broke on day one.").run()
    assert app.session_state.result is None
    assert any("changed since the last analysis" in i.value for i in app.info)
    button(app, "Clear").click().run()
    assert app.text_area(key="review").value == ""
    assert app.session_state.result is None


def test_invalid_review_keeps_text_and_records_nothing(app):
    app.text_area(key="review").input("12345 !!!").run()
    button(app, "Analyze review").click().run()
    assert not app.exception
    assert app.session_state.result["status"] == "unavailable"
    assert app.text_area(key="review").value == "12345 !!!"
    assert any("Not analyzed" in w.value and "two letters" in w.value for w in app.warning)
    assert not app.session_state.get("history")


def test_feedback_is_attached_to_history(app):
    analyze(app, LUKEWARM)
    app.radio(key="feedback_expected").set_value("negative").run()
    app.button(key="save_feedback").click().run()
    assert app.session_state.history[-1]["feedback"] == "Reported incorrect; expected negative"


def test_review_text_is_never_rendered_as_markdown(app):
    tricky = "Great [click here](http://example.com) **bold** product, works well"
    app.text_area(key="review").input(tricky).run()
    visit(app, "Explain")
    assert tricky in [t.value for t in app.text]
    assert not any("example.com" in m.value for m in app.markdown)


def test_history_page_details_delete_and_use_as_dataset(app):
    analyze(app, PRAISE)
    analyze(app, COMPLAINT)
    visit(app, "History")
    assert any("2** of 2 analyses" in m.value for m in app.markdown)
    app.selectbox(key="hist_detail").set_value(1).run()
    assert "Review row 1" in [h.value for h in app.header]
    button(app, "Delete this analysis").click().run()
    assert [e["sentiment"] for e in app.session_state.history] == ["negative"]
    button(app, "Use as dataset").click().run()
    assert app.session_state.page == "Overview"
    assert app.session_state.dataset_meta["kind"] == "history"
    visit(app, "Overview")
    assert app.session_state.dataset.sentiment.tolist() == ["negative"]
    visit(app, "History")
    button(app, "Yes, clear history").click().run()
    assert app.session_state.history == []


def test_csv_upload_download_and_invalid_upload(app):
    visit(app, "Batch Analysis")
    app.file_uploader[0].set_value(
        ("reviews.csv", b"review_text\ngreat product\nawful product\naverage product", "text/csv")
    ).run()
    button(app, "Analyze CSV").click().run()
    assert not app.exception
    assert len(app.session_state.bulk_result) == 3
    assert len(app.get("download_button")) == 2
    assert app.session_state.dataset_meta["name"] == "reviews.csv"
    assert any("3 of 3 input rows processed" in s.value for s in app.success)
    app.file_uploader[0].set_value(("invalid.csv", b"", "text/csv")).run()
    assert app.warning
    assert len(app.get("download_button")) == 1


def test_csv_with_metadata_flows_into_dashboard(app):
    visit(app, "Batch Analysis")
    content = (
        "id,review,stars,date,product\n"
        "a1,Absolutely love it,5,2026-01-01,Mug\n"
        "a2,Terrible and broke fast,1,2026-01-08,Mug\n"
        "a3,,3,2026-01-15,Lamp\n"
        "a4,It is okay I guess,3,not a date,Lamp\n"
        "a5,absolutely love it,4,2026-01-20,Lamp\n"
    ).encode()
    app.file_uploader[0].set_value(("shop.csv", content, "text/csv")).run()
    assert any("1 rows repeat an earlier review" in c.value for c in app.caption)
    button(app, "Analyze CSV").click().run()
    assert not app.exception
    frame = app.session_state.dataset
    assert frame.row.tolist() == [1, 2, 3, 4, 5]
    assert frame.row_id.tolist() == ["a1", "a2", "a3", "a4", "a5"]
    assert frame.status.tolist()[2] == "unavailable" and frame.message.iloc[2]
    assert frame.rating.tolist() == [5, 1, 3, 3, 4]
    assert frame.date.isna().tolist() == [False, False, False, True, False]
    assert any("4 analyzed + 1 unavailable = 5" in s.value for s in app.success)
    # The header must describe the dataset this same run just created, not the previous one.
    assert "Dataset: shop.csv · 5 rows" in html_text(app)
    button(app, "Open in Overview").click().run()
    assert app.session_state.page == "Overview"
    visit(app, "Overview")
    headers = [h.value for h in app.header]
    assert {"Sentiment distribution", "Sentiment over time", "Insights"} <= set(headers)
    assert "Data and model status" in headers
    assert any("Too few dated reviews" in w.value for w in app.warning)
    visit(app, "Review Explorer")
    app.multiselect(key="ex_labels").set_value(["unavailable"]).run()
    assert any("**1** matching reviews of 5" in m.value for m in app.markdown)
    app.selectbox(key="ex_detail").select_index(0).run()
    assert not app.exception
    assert any(w.value for w in app.warning)  # unavailable reason shown, not a fake label


def test_sample_dataset_explorer_search_and_paging(app):
    visit(app, "Overview")
    button(app, "Load sample dataset").click().run(timeout=120)
    assert not app.exception
    assert len(app.session_state.dataset) == 2698
    assert any("Trends need real review dates" in i.value for i in app.info)
    assert "Sentiment by product or category" in [h.value for h in app.header]
    visit(app, "Review Explorer")
    app.text_input(key="ex_query").set_value("battery").run()
    assert any("**25** matching" in m.value for m in app.markdown)
    app.text_input(key="ex_query").set_value("").run()
    button(app, "Next →").click().run()
    assert app.session_state.ex_page == 2
    visit(app, "History")
    visit(app, "Review Explorer")
    assert app.session_state.ex_page == 2  # filters survive page switches
    app.selectbox(key="ex_detail").select_index(0).run()
    assert not app.exception and len(app.code) == 1


def test_missing_reports_still_allows_prediction(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTIMENT_REPORTS_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run(timeout=30)
    assert not app.exception
    assert analyze(app, PRAISE)["label"] == "positive"
    visit(app, "Model & Evaluation")
    assert any("reports are unavailable" in info.value for info in app.info)
    assert "Model information" in [h.value for h in app.header]


def test_model_page_shows_saved_metrics(app):
    visit(app, "Model & Evaluation")
    labels = [m.label for m in app.metric]
    assert {"Held-out accuracy", "Macro F1", "Baseline accuracy"} <= set(labels)
    headers = [h.value for h in app.header]
    assert {"Decision rule evaluation", "Fast vs Deep Mode"} <= set(headers)


def test_report_mismatch_rejected():
    from src.ui import load_reports

    with pytest.raises(ValueError, match="different model"):
        load_reports(ROOT / "reports", {"artifact_sha256": "wrong"})


def test_app_never_imports_training():
    import ast

    for path in [ROOT / "app.py", *list((ROOT / "sentix").glob("*.py"))]:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        assert not any(
            isinstance(n, ast.ImportFrom) and n.module == "src.train" for n in ast.walk(tree)
        )
        assert not any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "fit"
            for n in ast.walk(tree)
        )
