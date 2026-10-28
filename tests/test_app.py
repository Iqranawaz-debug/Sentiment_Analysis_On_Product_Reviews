from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from sentix.views import PAGES

ROOT = Path(__file__).resolve().parents[1]
FALLBACK = "couldn't open this feature"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(ROOT / "models"))
    monkeypatch.setenv("SENTIMENT_REPORTS_DIR", str(ROOT / "reports"))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run(timeout=60)


def button(app, label):
    return next(b for b in app.button if b.label == label)


def visit(app, page):
    app.radio(key="page").set_value(page).run(timeout=60)
    assert not app.exception
    assert not any(FALLBACK in info.value for info in app.info), page
    return app


def test_missing_model_screen(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run(timeout=30)
    assert not app.exception
    assert "could not be loaded" in app.error[0].value


def test_every_page_renders_without_a_dataset(app):
    assert app.session_state.page == "Review Analyzer"
    assert app.radio(key="page").options == list(PAGES)
    for page in PAGES:
        visit(app, page)
    visit(app, "Overview")
    assert button(app, "Load sample dataset")
    assert button(app, "No analyzed reviews yet").disabled


def test_app_pages_and_predictions(app):
    assert not app.exception
    assert not app.error
    for label, expected in [
        ("😍 Love", "positive"),
        ("😡 Hate", "negative"),
        ("😐 Meh", "neutral"),
    ]:
        button(app, label).click().run()
        assert not app.exception
        assert app.session_state.result["label"] == expected
    assert [entry["sentiment"] for entry in app.session_state.history] == [
        "positive",
        "negative",
        "neutral",
    ]
    assert app.session_state.result_meta["elapsed_ms"] >= 0
    visit(app, "Explain")
    button(app, "Explain this review").click().run()
    assert not app.exception
    assert app.session_state.explanation[0] == app.session_state.review
    visit(app, "Guess the Mood")
    button(app, "Happy").click().run()
    assert not app.exception
    assert app.session_state.game_reveal is not None
    assert app.session_state.result["label"] == "neutral"
    button(app, "Reset score").click().run()
    assert app.session_state.game_score == [0, 0]
    visit(app, "Review Analyzer")
    assert app.text_area(key="review").value == app.session_state.analyzed_text


def test_invalid_review_keeps_text_and_records_nothing(app):
    app.text_area(key="review").input("12345 !!!").run()
    button(app, "Read the mood").click().run()
    assert not app.exception
    assert app.session_state.result["status"] == "unavailable"
    assert app.text_area(key="review").value == "12345 !!!"
    assert any("two letters" in info.value for info in app.info)
    assert not app.session_state.get("history")


def test_history_page_and_use_as_dataset(app):
    button(app, "😍 Love").click().run()
    visit(app, "History")
    assert any("1** of 1 analyses" in m.value for m in app.markdown)
    button(app, "Use as dataset").click().run()
    assert app.session_state.page == "Overview"
    assert app.session_state.dataset_meta["kind"] == "history"
    visit(app, "Overview")
    assert app.session_state.dataset.sentiment.tolist() == ["positive"]
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
    ).encode()
    app.file_uploader[0].set_value(("shop.csv", content, "text/csv")).run()
    button(app, "Analyze CSV").click().run()
    assert not app.exception
    frame = app.session_state.dataset
    assert frame.row.tolist() == [1, 2, 3, 4]
    assert frame.row_id.tolist() == ["a1", "a2", "a3", "a4"]
    assert frame.status.tolist()[2] == "unavailable" and frame.message.iloc[2]
    assert frame.rating.tolist() == [5, 1, 3, 3]
    assert frame.date.isna().tolist() == [False, False, False, True]
    # The sidebar must describe the dataset this same run just created, not the previous one.
    assert "Active dataset: shop.csv (4 rows)" in [c.value for c in app.sidebar.caption]
    button(app, "Open in Overview").click().run()
    assert app.session_state.page == "Overview"
    visit(app, "Overview")
    headers = [h.value for h in app.header]
    assert {"Sentiment distribution", "Sentiment over time", "Insights"} <= set(headers)
    assert any("Too few dated reviews" in w.value for w in app.warning)
    visit(app, "Review Explorer")
    app.multiselect(key="ex_labels").set_value(["unavailable"]).run()
    assert any("**1** matching reviews of 4" in m.value for m in app.markdown)
    app.selectbox(key="ex_detail").select_index(0).run()
    assert not app.exception
    assert any(w.value for w in app.warning)  # unavailable reason shown, not a fake label


def test_sample_dataset_explorer_search_and_paging(app):
    visit(app, "Overview")
    button(app, "Load sample dataset").click().run(timeout=120)
    assert not app.exception
    assert len(app.session_state.dataset) == 2698
    assert any("Trends need real review dates" in i.value for i in app.info)
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
    button(app, "😍 Love").click().run()
    assert app.session_state.result["label"] == "positive"
    visit(app, "Model & Evaluation")
    assert any("reports are unavailable" in info.value for info in app.info)
    assert "Model information" in [h.value for h in app.header]


def test_model_page_shows_saved_metrics(app):
    visit(app, "Model & Evaluation")
    labels = [m.label for m in app.metric]
    assert {"Held-out accuracy", "Macro F1", "Baseline accuracy"} <= set(labels)


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
