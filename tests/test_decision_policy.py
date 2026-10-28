"""Class-prior decision rule: bound to the artifact, never alters probabilities.

Expected labels below come from star-rating proxies or the project's curated
regression set; ambiguous sentences only assert that uncertainty is flagged.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from sentix.deep import DeepPredictor
from sentix.mascot import presentation_mood
from sentix.predictor import LABELS, Predictor, load_model, load_policy

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def predictor():
    return load_model()


def test_policy_is_bound_to_this_artifact(predictor, tmp_path):
    assert predictor.policy is not None and predictor.policy["adopted"]
    report = json.loads((ROOT / "reports/decision_policy.json").read_text(encoding="utf-8"))
    assert predictor.policy["alpha"] == report["alpha"]
    sha = predictor.metadata["artifact_sha256"]
    assert load_policy(ROOT / "models", "0" * 64) is None
    for broken in (
        "{",
        json.dumps({**predictor.policy, "adopted": False}),
        json.dumps({**predictor.policy, "weights": {"negative": -1}}),
    ):
        (tmp_path / "decision_policy.json").write_text(broken)
        assert load_policy(tmp_path, sha) is None
    assert load_policy(tmp_path / "missing", sha) is None


def test_probabilities_are_the_calibrated_pipeline_output(predictor):
    texts = ["It broke immediately", "It is average", "Great value and tastes fresh"]
    raw = predictor.model.predict_proba(texts)
    order = [list(predictor.model.classes_).index(label) for label in LABELS]
    for text, expected in zip(texts, raw[:, order]):
        result = predictor.predict(text)
        assert [result["probabilities"][k] for k in LABELS] == pytest.approx(expected)
        assert result["confidence"] == result["probabilities"][result["label"]]
        assert result["top_probability"] == max(result["probabilities"].values())


def test_without_policy_labels_are_plain_argmax(predictor):
    plain = Predictor(predictor.model, predictor.metadata)
    texts = ["It broke immediately", "It is average", "I love it"]
    assert [r["label"] for r in plain.predict_batch(texts)] == list(predictor.model.predict(texts))
    assert all("rule_changed_label" not in r["uncertainty"] for r in plain.predict_batch(texts))


def test_rule_cannot_change_confident_predictions(predictor):
    """Deep Mode's approved gate (top probability < 0.80) is therefore unaffected."""
    rng = np.random.default_rng(0)
    points = rng.dirichlet([0.3, 0.3, 0.3], 20000)
    confident = points[points.max(axis=1) >= 0.80]
    assert len(confident) > 1000
    assert ((confident * predictor.weights).argmax(axis=1) == confident.argmax(axis=1)).all()


@pytest.mark.parametrize(
    "text,label",
    [("It broke immediately", "negative"), ("It is average", "neutral")],
)
def test_curated_cases_fixed_by_rule_are_flagged_tentative(predictor, text, label):
    result = predictor.predict(text)
    assert result["label"] == label
    assert "rule_changed_label" in result["uncertainty"]
    assert result["top_label"] != label


@pytest.mark.parametrize("text", ["I hate it", "Absolutely hate it"])
def test_known_failures_are_shown_as_tentative(predictor, text):
    """Still labelled positive by the model (a documented failure); the interface must
    present it as tentative rather than as a confident verdict."""
    result = predictor.predict(text)
    assert result["label"] == "positive"
    assert presentation_mood(result) == "unsure"
    if text == "I hate it":
        assert "classifier_disagrees" in result["uncertainty"]


@pytest.mark.parametrize(
    "text,label",
    [
        ("Absolutely love this product, works perfectly and arrived early!", "positive"),
        ("Terrible quality. It broke in two days. Total waste of money.", "negative"),
        ("I wouldn't recommend it.", "negative"),
        ("It is not bad for the price.", "neutral"),
        ("Not good", "negative"),
        ("Does not work", "negative"),
        ("Highly recommend", "positive"),
    ],
)
def test_clear_cases_and_negation(predictor, text, label):
    result = predictor.predict(text)
    assert result["label"] == label
    assert set(result["probabilities"]) == set(LABELS)


def test_saved_regression_results_reproduce(predictor):
    report = json.loads((ROOT / "reports/decision_policy.json").read_text(encoding="utf-8"))
    cases = report["regression"]["cases"]
    labels = [r["label"] for r in predictor.predict_batch([c["review_text"] for c in cases])]
    assert labels == [c["after"] for c in cases]
    assert report["test"]["after"]["macro_f1"] > report["test"]["before"]["macro_f1"]


class FakeProduction:
    metadata = {
        "artifact_sha256": json.loads(
            (ROOT / "configs/deep_mode.json").read_text(encoding="utf-8")
        )["production_sha256"]
    }

    def predict_batch(self, texts):
        # Label chosen by the rule has low probability, but the top class is confident.
        return [
            {
                "status": "ok",
                "label": "neutral",
                "confidence": 0.15,
                "top_probability": 0.85,
                "probabilities": {"negative": 0.0, "neutral": 0.15, "positive": 0.85},
                "uncertainty": [],
            }
            for _ in texts
        ]


class ExplodingRuntime:
    def predict_proba(self, texts, progress=None):
        raise AssertionError("confident rows must not be deferred")


def test_deep_gate_uses_top_probability():
    result = DeepPredictor(FakeProduction(), runtime=ExplodingRuntime()).predict("anything")
    assert result["route"] == "fast"
