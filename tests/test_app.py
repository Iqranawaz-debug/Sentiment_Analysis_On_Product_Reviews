from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_missing_model_screen(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert "could not be loaded" in app.error[0].value


def test_app_pages_and_predictions():
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert not app.error
    for page in ["Data exploration", "Model performance", "Explainability", "About and model info", "Batch CSV", "Single review"]:
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception, page
    app.button[0].click().run()
    assert "non-empty" in app.error[0].value
    app.text_area[0].set_value("Excellent quality and great value.")
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert "Predicted sentiment" in app.success[0].value
    app.text_area[0].set_value("😀!!!")
    app.button[0].click().run()
    assert app.error
    app.text_area[0].set_value("good " * 5000)
    app.button[0].click().run()
    assert not app.exception
    assert len(app.text_area[0].value) <= 20000


def test_csv_upload_download_and_invalid_upload():
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    app.sidebar.radio[0].set_value("Batch CSV").run()
    app.file_uploader[0].set_value(("reviews.csv", (ROOT / "data/example_reviews.csv").read_bytes(), "text/csv")).run()
    assert not app.exception
    assert not app.error
    assert len(app.dataframe[0].value) == 3
    assert list(app.dataframe[0].value.sentiment) == ["positive", "negative", "neutral"]
    assert len(app.get("download_button")) == 2
    app.get("download_button")[1].click().run()
    assert not app.exception
    app.file_uploader[0].set_value(("invalid.csv", b"wrong\ngreat", "text/csv")).run()
    assert "review_text column" in app.error[0].value
    assert len(app.get("download_button")) == 1
