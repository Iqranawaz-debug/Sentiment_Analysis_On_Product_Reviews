from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(ROOT / "models"))
    monkeypatch.setenv("SENTIMENT_REPORTS_DIR", str(ROOT / "reports"))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run(timeout=30)


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_missing_model_screen(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run(timeout=30)
    assert not app.exception
    assert "could not be loaded" in app.error[0].value


def test_app_tabs_and_predictions(app):
    assert not app.exception
    assert not app.error
    assert len(app.tabs) == 5
    for label, expected in [
        ("😍 Love", "positive"),
        ("😡 Hate", "negative"),
        ("😐 Meh", "neutral"),
    ]:
        button(app, label).click().run()
        assert not app.exception
        assert app.session_state.result["label"] == expected
    button(app, "Explain this review").click().run()
    assert not app.exception
    assert app.session_state.explanation[0] == app.session_state.review
    button(app, "Happy").click().run()
    assert not app.exception
    assert app.session_state.game_reveal is not None
    assert app.session_state.result["label"] == "neutral"
    button(app, "Reset score").click().run()
    assert app.session_state.game_score == [0, 0]


def test_csv_upload_download_and_invalid_upload(app):
    app.file_uploader[0].set_value(
        ("reviews.csv", b"review_text\ngreat product\nawful product\naverage product", "text/csv")
    ).run()
    button(app, "Analyze CSV").click().run()
    assert not app.exception
    assert len(app.session_state.bulk_result) == 3
    assert len(app.tabs[1].get("download_button")) == 2
    app.file_uploader[0].set_value(("invalid.csv", b"", "text/csv")).run()
    assert app.warning
    assert len(app.tabs[1].get("download_button")) == 1


def test_missing_reports_still_allows_prediction(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTIMENT_REPORTS_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run(timeout=30)
    assert not app.exception
    assert any("reports are unavailable" in info.value for info in app.info)
    button(app, "😍 Love").click().run()
    assert app.session_state.result["label"] == "positive"


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
