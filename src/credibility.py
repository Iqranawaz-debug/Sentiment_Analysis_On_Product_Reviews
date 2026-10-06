"""Seeded paired-row bootstrap reporting; never invokes a model."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import LABELS


def bootstrap_intervals(labels, predictions, seed=42, resamples=1000):
    """Multinomial counts are exactly the IID paired-row bootstrap distribution.

    Keep all label axes even when a replicate omits a class. These percentile
    intervals are conditional on the fitted model, not training/selection uncertainty.
    """
    positions = {label: i for i, label in enumerate(LABELS)}
    cells = np.array(
        [positions[a] * len(LABELS) + positions[b] for a, b in zip(labels, predictions)]
    )
    if not len(cells):
        raise ValueError("Cannot bootstrap an empty population.")
    counts = np.bincount(cells, minlength=len(LABELS) ** 2)
    matrices = (
        np.random.default_rng(seed)
        .multinomial(len(cells), counts / len(cells), size=resamples)
        .reshape(-1, 3, 3)
    )
    true_positive = np.diagonal(matrices, axis1=1, axis2=2)
    denominators = matrices.sum(axis=1) + matrices.sum(axis=2)
    f1 = np.divide(
        2 * true_positive,
        denominators,
        out=np.zeros_like(true_positive, dtype=float),
        where=denominators != 0,
    ).mean(axis=1)
    accuracy = true_positive.sum(axis=1) / len(cells)
    return {
        "method": "IID paired-row percentile bootstrap (multinomial confusion counts)",
        "confidence_level": 0.95,
        "resamples": resamples,
        "seed": seed,
        "accuracy": dict(zip(("lower", "upper"), np.quantile(accuracy, [0.025, 0.975]).tolist())),
        "macro_f1": dict(zip(("lower", "upper"), np.quantile(f1, [0.025, 0.975]).tolist())),
    }


def credible_metrics(labels, predictions, majority_label, seed=42, resamples=1000, min_support=30):
    from src.evaluate import evaluate_predictions

    labels, predictions = np.asarray(labels), np.asarray(predictions)
    result = evaluate_predictions(labels, predictions)
    result["class_support"] = {label: int((labels == label).sum()) for label in LABELS}
    result["low_support_classes"] = [
        label for label, n in result["class_support"].items() if n < min_support
    ]
    result["low_support"] = bool(result["low_support_classes"])
    result["low_support_threshold"] = min_support
    result["confidence_intervals"] = bootstrap_intervals(labels, predictions, seed, resamples)
    baseline = np.repeat(majority_label, len(labels))
    result["majority_baseline"] = evaluate_predictions(labels, baseline)
    result["majority_baseline"]["label_from_training"] = majority_label
    return result


def add_credibility(metrics, rows, config=None):
    cfg = config or {"resamples": 1000, "min_class_support": 30}
    majority = max(
        metrics["split_class_distribution"]["train"],
        key=metrics["split_class_distribution"]["train"].get,
    )

    def summarize(frame):
        return credible_metrics(
            frame.true_label,
            frame.predicted_label,
            majority,
            metrics["seed"],
            cfg["resamples"],
            cfg["min_class_support"],
        )

    metrics["test"] = summarize(rows)
    metrics["per_source"] = {
        source: summarize(part) for source, part in rows.groupby("source", sort=True)
    }
    metrics["majority_baseline"] = metrics["test"]["majority_baseline"]
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()
    path = args.reports / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.read_csv(args.reports / "predictions.csv")
    add_credibility(metrics, rows)
    path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    metadata_path = args.models / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["artifact_sha256"] != metrics["artifact_sha256"]:
        raise ValueError("Reports and model metadata do not match.")
    metadata["metrics"] = metrics["test"]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
