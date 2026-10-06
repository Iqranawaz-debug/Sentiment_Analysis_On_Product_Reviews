import numpy as np

from src.credibility import bootstrap_intervals, credible_metrics


def test_bootstrap_deterministic_and_perfect():
    labels = np.array(["negative", "neutral", "positive"] * 50)
    first = bootstrap_intervals(labels, labels)
    assert first == bootstrap_intervals(labels, labels)
    assert first["resamples"] == 1000
    assert first["accuracy"] == {"lower": 1.0, "upper": 1.0}
    assert first["macro_f1"] == {"lower": 1.0, "upper": 1.0}


def test_low_support_includes_absent_classes_and_training_majority():
    labels = ["positive"] * 30 + ["neutral"] * 5
    result = credible_metrics(labels, labels, "negative")
    assert result["low_support_classes"] == ["negative", "neutral"]
    assert result["class_support"] == {"negative": 0, "neutral": 5, "positive": 30}
    assert result["majority_baseline"]["accuracy"] == 0
    assert result["majority_baseline"]["label_from_training"] == "negative"
    for metric in ("accuracy", "macro_f1"):
        ci = result["confidence_intervals"][metric]
        assert 0 <= ci["lower"] <= ci["upper"] <= 1
