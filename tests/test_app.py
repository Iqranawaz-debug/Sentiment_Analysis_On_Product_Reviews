from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.config import load_config
from src.train import train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("app")
    source = directory / "synthetic.csv"
    pd.DataFrame(
        [
            {"review_text": f"{word} product {i}", "sentiment": label}
            for label, word in [
                ("negative", "awful"),
                ("neutral", "average"),
                ("positive", "great"),
            ]
            for i in range(60)
        ]
    ).to_csv(source, index=False)
    config = load_config()
    config["features"] = {"min_df": 1, "max_features": 100, "char_max_features": 100}
    config["paths"]["reports"] = str(directory / "reports")
    train([source], config, directory / "models")
    return directory


@pytest.fixture
def app(monkeypatch, app_artifact):
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(app_artifact / "models"))
    monkeypatch.setenv("SENTIMENT_REPORTS_DIR", str(app_artifact / "reports"))
    return AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)


def test_missing_model_screen(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert "could not be loaded" in app.error[0].value


def test_app_tabs_and_predictions(app):
    assert not app.exception
    assert not app.error
    assert [tab.label for tab in app.tabs] == [
        "Predict",
        "Batch",
        "Model Comparison",
        "Error Explorer",
        "Insights",
        "About",
    ]
    app.button[0].click().run()
    assert "non-empty" in app.error[0].value
    app.text_area[0].set_value("great product")
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert "Predicted sentiment" in app.success[0].value
    app.text_area[0].set_value("😀!!!")
    app.button[0].click().run()
    assert app.error


def test_csv_upload_download_and_invalid_upload(app):
    app.file_uploader[0].set_value(
        ("reviews.csv", b"review_text\ngreat product\nawful product\naverage product", "text/csv")
    ).run()
    assert not app.exception
    assert not app.error
    assert len(app.tabs[1].dataframe[0].value) == 3
    assert "low_confidence" in app.tabs[1].dataframe[0].value.columns
    assert len(app.tabs[1].get("download_button")) == 2
    app.file_uploader[0].set_value(("invalid.csv", b"wrong\ngreat", "text/csv")).run()
    assert "review_text column" in app.error[0].value
    assert len(app.tabs[1].get("download_button")) == 1


def test_missing_reports_still_allows_prediction(monkeypatch, app_artifact, tmp_path):
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(app_artifact / "models"))
    monkeypatch.setenv("SENTIMENT_REPORTS_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert any("reports are unavailable" in info.value for info in app.info)
    app.text_area[0].set_value("great product")
    app.button[0].click().run()
    assert app.success


def test_report_mismatch_rejected(app_artifact):
    from src.ui import load_reports

    with pytest.raises(ValueError, match="different model"):
        load_reports(app_artifact / "reports", {"artifact_sha256": "wrong"})


def test_app_never_imports_training():
    import ast

    for filename in ("app.py", "src/ui.py", "src/predict.py"):
        tree = ast.parse((ROOT / filename).read_text(encoding="utf-8-sig"))
        assert not any(
            isinstance(node, ast.ImportFrom) and node.module == "src.train"
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
            for node in ast.walk(tree)
        )
