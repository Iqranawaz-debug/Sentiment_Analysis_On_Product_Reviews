"""Integration tests against the preserved saved model, never retraining."""

import pytest

from sentix.bulk import analyze, mood_score, read_csv
from sentix.explain import explain, highlight
from sentix.predictor import LABELS, load_model
from src.predict import export_csv


@pytest.fixture(scope="module")
def predictor():
    return load_model()


@pytest.mark.parametrize(
    "text,label",
    [
        ("Absolutely love this product, works perfectly and arrived early!", "positive"),
        ("Terrible quality. It broke in two days. Total waste of money.", "negative"),
        ("It is okay. Does the job, nothing special.", "neutral"),
    ],
)
def test_real_predictions(predictor, text, label):
    result = predictor.predict(text)
    assert result["label"] == label
    assert set(result["probabilities"]) == set(LABELS)
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["confidence"] == result["probabilities"][label]
    assert all(0 <= p <= 1 for p in result["probabilities"].values())


@pytest.mark.parametrize("text", ["", "   ", "😀!!!", "qzxvvv", "great " * 4000])
def test_invalid_inputs_are_honest(predictor, text):
    result = predictor.predict(text)
    assert result["label"] == "unavailable"
    assert result["confidence"] is None
    assert all(p is None for p in result["probabilities"].values())


def test_batch_and_export(predictor):
    frame, truncated = read_csv(b'Text\n"I love this"\n"Terrible waste of money"\n""\nextra', 3)
    assert truncated
    results = analyze(predictor, frame.Text.tolist())
    assert len(results) == 3
    assert results.iloc[2].status == "unavailable"
    assert mood_score(results) == 50
    assert b"probability_positive" in export_csv(results)
    assert predictor.predict_batch([]) == []


def test_explanation_signs_and_escaping(predictor):
    values = dict(explain("I love this but it broke", predictor))
    assert values["love"] > 0
    assert values["broke"] < 0
    assert "<script>" not in highlight("<script>alert(1)</script> love", list(values.items()))
