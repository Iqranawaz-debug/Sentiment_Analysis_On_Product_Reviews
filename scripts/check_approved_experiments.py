"""Verify saved evidence and report probability calibration without retuning."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from scripts.diagnose_short_reviews import CASES
from scripts.run_approved_experiments import (
    LABELS,
    MODELS,
    ROOT,
    RUN,
    clean_text,
    guard,
    probabilities,
    sha256_file,
    write,
)


def calibration_scores(labels, p):
    y = np.array([LABELS.index(label) for label in labels])
    target = np.eye(3)[y]
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    ece = 0.0
    for low in np.arange(0, 1, 0.1):
        mask = (confidence >= low) & (confidence < low + 0.1 if low < 0.9 else confidence <= 1)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "multiclass_brier_sum": float(np.mean(np.sum((p - target) ** 2, axis=1))),
        "ece_10_bins": float(ece),
    }


def main():
    guard()
    protocol = json.loads((RUN / "protocol.json").read_text())
    selection = json.loads((RUN / "selection.json").read_text())
    parts = {k: pd.read_csv(RUN / f"{k}.csv") for k in protocol["split_sha256"]}
    regression = {clean_text(t) for items in CASES.values() for t in items}
    for name, frame in parts.items():
        assert sha256_file(RUN / f"{name}.csv") == protocol["split_sha256"][name]
        assert not set(frame.normalized) & regression
        for other, other_frame in parts.items():
            if name != other:
                assert not set(frame.group) & set(other_frame.group)
                assert not set(frame.normalized) & set(other_frame.normalized)
    for row in selection["candidates"]:
        assert sha256_file(MODELS / row["name"] / "pipeline.joblib") == row["sha256"]
    rows = json.loads((RUN / "comparison.json").read_text())
    if (RUN / "transformer_results.json").exists():
        rows += json.loads((RUN / "transformer_results.json").read_text())
    calibration = []
    for row in rows:
        if row["name"] == "transformer":
            assert sha256_file(MODELS / "transformer_int8/state_dict.pt") == row["sha256"]
            source = json.loads((MODELS / "transformer/training_config.json").read_text())
            for artifact in source["files"]:
                assert sha256_file(MODELS / "transformer" / artifact["name"]) == artifact["sha256"]
        if row["name"].startswith("hybrid_"):
            assert sha256_file(MODELS / row["name"] / "training_config.json") == row["sha256"]
        for suffix in ["test", "regression"]:
            saved = pd.read_csv(RUN / f"{row['name']}_{suffix}_predictions.csv")
            p = saved[LABELS].to_numpy()
            if row["name"].startswith("hybrid_"):
                base = pd.read_csv(RUN / f"{row['tfidf']}_{suffix}_predictions.csv")[
                    LABELS
                ].to_numpy()
                fallback = pd.read_csv(RUN / f"transformer_{suffix}_predictions.csv")[
                    LABELS
                ].to_numpy()
                expected = np.where((base.max(axis=1) >= 0.80)[:, None], base, fallback)
                assert np.allclose(p, expected)
            assert np.allclose(p.sum(axis=1), 1, atol=1e-6)
            assert np.all((p >= 0) & (p <= 1))
            assert (saved.predicted.to_numpy() == np.array(LABELS)[p.argmax(axis=1)]).all()
            assert np.isclose((saved.predicted == saved.label).mean(), row[suffix]["accuracy"])
            assert sum(map(sum, row[suffix]["confusion_matrix"])) == len(saved)
            calibration.append(
                {"name": row["name"], "split": suffix, **calibration_scores(saved.label, p)}
            )
        if row["name"] == "production" or row["name"] in [
            r["name"] for r in selection["candidates"]
        ]:
            path = (
                ROOT / "models/pipeline.joblib"
                if row["name"] == "production"
                else MODELS / row["name"] / "pipeline.joblib"
            )
            model = joblib.load(path)
            val = parts["validation"]
            p = probabilities(model, val.review_text)
            calibration.append(
                {"name": row["name"], "split": "validation", **calibration_scores(val.label, p)}
            )
    write(RUN / "calibration_diagnostics.json", calibration)
    write(
        RUN / "verification.json",
        {
            "production_unchanged": True,
            "split_hashes_valid": True,
            "disjoint_groups_and_text": True,
            "regression_excluded": True,
            "candidate_hashes_valid": True,
            "prediction_probabilities_and_metrics_consistent": True,
            "models_checked": len(rows),
        },
    )
    print(
        "Verified",
        len(rows),
        "models/policies; split isolation, artifact hashes and saved predictions.",
    )


if __name__ == "__main__":
    main()
