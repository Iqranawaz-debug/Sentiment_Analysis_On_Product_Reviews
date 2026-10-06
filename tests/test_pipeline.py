import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.data import load_data, prepare_data
from src.predict import (
    MAX_BATCH_ROWS,
    MAX_CSV_BYTES,
    explain_review,
    export_csv,
    load_artifact,
    predict_reviews,
    read_batch,
)
from src.preprocessing import MAX_REVIEW_CHARS, clean_text, validate_review
from src.train import build_pipeline, train


@pytest.fixture
def toy_model():
    # Synthetic examples test contracts only; never used for published metrics.
    texts = [
        "awful broken bad",
        "bad poor terrible",
        "average ordinary okay",
        "okay mixed average",
        "excellent great good",
        "good wonderful excellent",
    ]
    labels = ["negative", "negative", "neutral", "neutral", "positive", "positive"]
    return build_pipeline(LogisticRegression(), {"min_df": 1, "max_features": 100}).fit(
        texts, labels
    )


def test_negation_html_unicode():
    assert clean_text("<b>It isn't GOOD!</b> &amp; can't work") == "it is not good can not work"
    assert "café" in clean_text("CAFÉ ☕")
    assert clean_text(clean_text("Won't <br> work!")) == clean_text("Won't <br> work!")


@pytest.mark.parametrize(
    "text",
    [None, "", "  ", "😀!!!", "1234", "a" * (MAX_REVIEW_CHARS + 1)],
    ids=["none", "empty", "whitespace", "emoji", "numeric", "long"],
)
def test_invalid_review(text):
    with pytest.raises(ValueError):
        validate_review(text)


def test_probability_order_and_preprocessing(toy_model, tmp_path):
    text = "<b>Excellent!</b>"
    result = predict_reviews(toy_model, [text]).iloc[0]
    for index, label in enumerate(toy_model.classes_):
        assert result[f"probability_{label}"] == pytest.approx(
            toy_model.predict_proba([text])[0, index]
        )
    assert np.allclose(toy_model.predict_proba([text]), toy_model.predict_proba([clean_text(text)]))
    joblib.dump(toy_model, tmp_path / "model.joblib")
    assert np.allclose(
        joblib.load(tmp_path / "model.joblib").predict_proba([text]),
        toy_model.predict_proba([text]),
    )


def test_unknown_terms_abstain(toy_model):
    row = predict_reviews(toy_model, ["zzqxw unknownword"]).iloc[0]
    assert row.sentiment == "unavailable"
    assert pd.isna(row.model_probability)


def test_explanation_matches_logit(toy_model):
    text = "excellent good"
    index = list(toy_model.classes_).index("positive")
    explanation = explain_review(toy_model, text, "positive", limit=100)
    assert explanation.contribution.sum() + toy_model.named_steps["clf"].intercept_[
        index
    ] == pytest.approx(toy_model.decision_function([text])[0, index])


def test_svm_scores_are_not_probabilities():
    model = build_pipeline(LinearSVC(), {"min_df": 1, "max_features": 100}).fit(
        ["awful bad", "average okay", "great good"], ["negative", "neutral", "positive"]
    )
    result = predict_reviews(model, ["great"])
    assert "margin_positive" in result
    assert "model_probability" not in result


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"wrong\ngood",
        b'review_text\n"unfinished',
        b"review_text\n",
        b"review_text\n\xff",
        b'review_text\n" "',
        b"x" * (MAX_CSV_BYTES + 1),
        b"review_text\n" + b"good\n" * (MAX_BATCH_ROWS + 1),
    ],
    ids=["empty", "schema", "quotes", "header_only", "encoding", "blank", "bytes", "rows"],
)
def test_invalid_csv(content):
    with pytest.raises(ValueError):
        read_batch(content)


def test_csv_prediction_download(toy_model):
    frame = read_batch(b'\xef\xbb\xbfreview_text,extra\n"great, good",ignore\n"awful bad",ignore\n')
    result = predict_reviews(toy_model, frame.review_text)
    downloaded = pd.read_csv(io.BytesIO(export_csv(result)))
    assert len(downloaded) == 2
    assert downloaded.sentiment.tolist() == result.sentiment.tolist()
    assert "extra" not in downloaded
    escaped = export_csv(
        pd.DataFrame({"review_text": ["=SUM(1,2)", " +command", "@formula"]})
    ).decode("utf-8-sig")
    assert "'=SUM" in escaped and "' +command" in escaped and "'@formula" in escaped


def test_duplicate_conflicting_and_missing_data():
    frame = pd.DataFrame(
        {
            "review_text": ["good", "GOOD!", "bad", "okay", "conflict", "CONFLICT", None, "!!!"],
            "label": [
                "positive",
                "positive",
                "negative",
                "neutral",
                "positive",
                "negative",
                "positive",
                "neutral",
            ],
        }
    )
    cleaned, audit = prepare_data(frame)
    assert len(cleaned) == 3
    assert audit["duplicate_rows_removed"] == 1
    assert audit["conflicting_rows_removed"] == 2
    assert audit["missing_rows_removed"] == 1
    assert audit["invalid_text_rows_removed"] == 1


@pytest.mark.parametrize("score", [0, 6, 2.5, "invalid"])
def test_rating_validation(tmp_path, score):
    file = tmp_path / "ratings.csv"
    pd.DataFrame({"Text": ["a review"], "Score": [score]}).to_csv(file, index=False)
    with pytest.raises(ValueError):
        load_data([file])


def test_training_artifact_contract(tmp_path):
    source = tmp_path / "synthetic.csv"
    rows = [
        {"review_text": f"{word} product example {index}", "sentiment": label}
        for label, word in [("negative", "awful"), ("neutral", "average"), ("positive", "great")]
        for index in range(60)
    ]
    pd.DataFrame(rows).to_csv(source, index=False)
    from src.config import load_config

    config = load_config()
    config["features"] = {"min_df": 1, "max_features": 100, "char_max_features": 100}
    config["paths"]["reports"] = str(tmp_path / "reports")
    output = tmp_path / "artifacts"
    metadata = train([source], config, output)
    model, stored = load_artifact(output)
    assert json.loads(json.dumps(metadata)) == stored
    reports = json.loads((tmp_path / "reports/metrics.json").read_text())
    assert sum(reports["split_counts"][key] for key in ("train", "val", "test")) == 180
    assert len(reports["comparison"]) == 3
    assert len(predict_reviews(model, ["great product"])) == 1
    from src.evaluate import evaluate_predictions
    from src.explain import linear_estimator

    saved = pd.read_csv(tmp_path / "reports/predictions.csv")
    for key, value in evaluate_predictions(saved.true_label, saved.predicted_label).items():
        assert value == reports["test"][key]
    assert sum(map(sum, reports["test"]["confusion_matrix"])) == len(saved)
    assert np.allclose(model.predict_proba(["great product"]).sum(axis=1), 1)
    base = linear_estimator(model)
    if hasattr(base, "coef_"):
        index = list(base.classes_).index("positive")
        parts = explain_review(model, "great product", "positive", limit=100)
        score = base.decision_function(model.named_steps["tfidf"].transform(["great product"]))
        assert parts.contribution.sum() + base.intercept_[index] == pytest.approx(score[0, index])
    with pytest.raises(ValueError, match="Reports already exist"):
        train([source], config, output)
    (output / "pipeline.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(output)


def test_missing_artifact(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        load_artifact(tmp_path)


def test_prediction_schema_and_threshold(toy_model):
    from src.predict import predict

    result = predict(toy_model, "good product", threshold=1.0)
    assert set(result) == {"label", "probabilities", "low_confidence"}
    assert set(result["probabilities"]) == {"negative", "neutral", "positive"}
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["low_confidence"] is True
    assert predict(toy_model, "good product", threshold=0)["low_confidence"] is False
    unknown = predict(toy_model, "zzqxword")
    assert unknown["label"] == "unavailable"
    assert unknown["low_confidence"] is True
    assert all(value is None for value in unknown["probabilities"].values())


def test_committed_artifact():
    directory = Path(__file__).resolve().parents[1] / "models"
    if not (directory / "pipeline.joblib").exists():
        pytest.skip("No real model produced yet")
    model, metadata = load_artifact(directory)
    assert metadata["schema_version"] in (1, 2)
    assert set(predict_reviews(model, ["The product is excellent"]).sentiment) <= {
        "negative",
        "neutral",
        "positive",
    }
