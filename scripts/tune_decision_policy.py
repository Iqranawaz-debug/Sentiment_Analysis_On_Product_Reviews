"""Tune a class-prior decision rule for the production artifact on validation only.

Why: the sigmoid one-vs-rest calibration re-introduces the training class prior
(78% positive), so argmax of calibrated probabilities flips some clearly negative
short reviews (for example "I hate it") to positive even though the underlying
logistic regression ranks negative first. The rule divides calibrated
probabilities by training priors raised to alpha before taking the argmax:

    label = argmax_c  p_c / prior_c ** alpha

alpha = 0 is the current behaviour; alpha = 1 is full prior correction.
Probabilities themselves are never changed.

Protocol (fixed before looking at test results):
- Data: reports/approved_20261016 splits, which exclude all 30,000 rows of the
  production sample and near-duplicates of them, so they are held out from the
  production model. Labels are star-rating proxies.
- alpha is chosen on the validation split only: highest macro-F1 among values
  whose validation accuracy is within 3 points of alpha = 0.
- The rule is adopted only if validation macro-F1 improves by at least 0.01.
- The test split and the 24-review regression set are scored once afterwards.

Run:  python -m scripts.tune_decision_policy
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from sentix.predictor import LABELS, Predictor, load_model

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "reports/approved_20261016"
ALPHAS = [round(a, 2) for a in np.arange(0, 1.0001, 0.05)]
MAX_ACCURACY_DROP = 0.03
MIN_GAIN = 0.01


def probabilities(model, texts: list[str]) -> np.ndarray:
    raw = model.predict_proba(texts)
    order = [list(model.classes_).index(label) for label in LABELS]
    return raw[:, order]


def decide(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.array(LABELS)[(probs * weights).argmax(axis=1)]


def scores(truth, predicted) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(LABELS), zero_division=0
    )
    return {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=list(LABELS), average="macro")),
        "weighted_f1": float(f1_score(truth, predicted, labels=list(LABELS), average="weighted")),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(LABELS)).tolist(),
    }


def bootstrap(truth, before, after, seed=20261028, resamples=1000) -> dict:
    """Paired percentile intervals for the change in accuracy and macro-F1."""
    rng = np.random.default_rng(seed)
    truth, before, after = map(np.asarray, (truth, before, after))
    deltas = {"accuracy": [], "macro_f1": []}
    for _ in range(resamples):
        i = rng.integers(0, len(truth), len(truth))
        deltas["accuracy"].append(
            accuracy_score(truth[i], after[i]) - accuracy_score(truth[i], before[i])
        )
        deltas["macro_f1"].append(
            f1_score(truth[i], after[i], labels=list(LABELS), average="macro")
            - f1_score(truth[i], before[i], labels=list(LABELS), average="macro")
        )
    return {
        k: {"lower": float(np.percentile(v, 2.5)), "upper": float(np.percentile(v, 97.5))}
        for k, v in deltas.items()
    } | {"resamples": resamples, "seed": seed}


def main() -> None:
    predictor = load_model()
    model, metadata = predictor.model, predictor.metadata
    metrics = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))
    train = metrics["split_class_distribution"]["train"]
    priors = np.array([train[label] / sum(train.values()) for label in LABELS])

    validation = pd.read_csv(SPLITS / "validation.csv")
    test = pd.read_csv(SPLITS / "test.csv")
    regression = pd.read_csv(SPLITS / "production_regression_predictions.csv")
    val_probs = probabilities(model, validation.review_text.tolist())

    grid = []
    for alpha in ALPHAS:
        predicted = decide(val_probs, priors**-alpha)
        grid.append(
            {
                "alpha": alpha,
                "accuracy": float(accuracy_score(validation.label, predicted)),
                "macro_f1": float(
                    f1_score(validation.label, predicted, labels=list(LABELS), average="macro")
                ),
            }
        )
    base = grid[0]
    eligible = [g for g in grid if g["accuracy"] >= base["accuracy"] - MAX_ACCURACY_DROP]
    chosen = max(eligible, key=lambda g: (g["macro_f1"], -g["alpha"]))
    adopted = chosen["macro_f1"] - base["macro_f1"] >= MIN_GAIN
    alpha = chosen["alpha"] if adopted else 0.0
    weights = priors**-alpha

    # Single post-selection pass over test and the regression set.
    results = {}
    for name, frame in (("test", test), ("regression", regression)):
        probs = probabilities(model, frame.review_text.tolist())
        before, after = decide(probs, np.ones(3)), decide(probs, weights)
        results[name] = {
            "before": scores(frame.label, before),
            "after": scores(frame.label, after),
            "changed_predictions": int((before != after).sum()),
        }
        if name == "test":
            results[name]["paired_bootstrap_delta"] = bootstrap(frame.label, before, after)
            short = frame.review_text.str.split().str.len() <= 8
            results[name]["short_reviews"] = {
                "before": scores(frame.label[short], before[short.to_numpy()]),
                "after": scores(frame.label[short], after[short.to_numpy()]),
            }
        else:
            results[name]["cases"] = [
                {"review_text": t, "label": y, "before": b, "after": a}
                for t, y, b, a in zip(frame.review_text, frame.label, before, after)
            ]

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    policy = {
        "schema_version": 1,
        "artifact_sha256": metadata["artifact_sha256"],
        "rule": "argmax(p_c / prior_c ** alpha)",
        "alpha": alpha,
        "priors": dict(zip(LABELS, priors.round(6).tolist())),
        "weights": dict(zip(LABELS, weights.round(6).tolist())),
        "adopted": adopted,
        "selected_on": "reports/approved_20261016/validation.csv",
        "created_utc": created,
    }
    # The same rule through the app's predictor: rows the app marks unavailable (no
    # known vocabulary) count as wrong, matching scripts/verify_deep_mode.py.
    app = Predictor(model, metadata, policy=policy if adopted else None)
    for name, frame in (("test", test), ("regression", regression)):
        live = [r["label"] for r in app.predict_batch(frame.review_text.tolist())]
        results[name]["app"] = scores(frame.label, np.array(live)) | {
            "unavailable_counted_incorrect": live.count("unavailable")
        }
    report = {
        **policy,
        "protocol": __doc__.split("Protocol (fixed before looking at test results):")[1]
        .split("Run:")[0]
        .strip(),
        "validation_grid": grid,
        "validation_chosen": chosen,
        **results,
    }
    (ROOT / "models/decision_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    (ROOT / "reports/decision_policy.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"alpha": alpha, "adopted": adopted, "validation": chosen, "validation_base": base},
            indent=2,
        )
    )
    for name in results:
        b, a = results[name]["before"], results[name]["after"]
        print(
            f"{name}: acc {b['accuracy']:.4f} -> {a['accuracy']:.4f}; "
            f"macro-F1 {b['macro_f1']:.4f} -> {a['macro_f1']:.4f}; "
            f"changed {results[name]['changed_predictions']}"
        )
        print(
            "  recall",
            {k: round(b["per_class"][k]["recall"], 3) for k in LABELS},
            "->",
            {k: round(a["per_class"][k]["recall"], 3) for k in LABELS},
        )


if __name__ == "__main__":
    main()
