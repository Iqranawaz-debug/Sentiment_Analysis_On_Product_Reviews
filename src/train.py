"""Offline selection, validation tuning, calibration and one final test pass."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from src.config import load_config
from src.credibility import add_credibility
from src.data import load_data, sha256_file, split_data
from src.evaluate import evaluate_predictions, write_reports
from src.features import build_pipeline

LOG = logging.getLogger(__name__)


def estimator(name, seed, c=1.0):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logistic":
        return LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed)
    return LinearSVC(C=c, class_weight="balanced", max_iter=5000, random_state=seed)


def train(paths, config, output=None, max_samples=None, transformer=False):
    output = Path(output or config["paths"]["models"])
    reports = Path(config["paths"]["reports"])
    # Prevent accidentally replacing the evidence for a previously inspected test set.
    if (reports / "metrics.json").exists():
        raise ValueError(
            "Reports already exist. Use a fresh reports/models directory for a new experiment."
        )
    frame, audit = load_data(paths)
    train_df, val, test = split_data(frame, config["seed"], max_samples)
    fit, calibration = train_test_split(
        train_df,
        test_size=config["calibration_fraction"],
        stratify=train_df.label,
        random_state=config["seed"],
    )
    if fit.label.value_counts().min() < config["cv_folds"]:
        raise ValueError("Too few training examples per class for five-fold CV.")
    cv = StratifiedKFold(config["cv_folds"], shuffle=True, random_state=config["seed"])
    comparison = []
    for name in ("majority", "logistic", "linear_svc"):
        LOG.info("Five-fold training CV: %s", name)
        pipeline = build_pipeline(
            estimator(name, config["seed"]), config["features"], char=name == "linear_svc"
        )
        scores = cross_val_score(
            pipeline,
            fit.review_text,
            fit.label,
            cv=cv,
            scoring="f1_macro",
            n_jobs=config["n_jobs"],
            error_score="raise",
        )
        comparison.append(
            {
                "model": name,
                "cv_macro_f1": float(scores.mean()),
                "cv_macro_f1_std": float(scores.std()),
                "cv_folds": len(scores),
                "cv_fold_scores": scores.tolist(),
            }
        )
    winner = max(comparison, key=lambda row: row["cv_macro_f1"])["model"]
    LOG.info("Selected family: %s; tuning only on validation", winner)
    tuning, best, best_score = [], None, -1.0
    for c in config["model_grids"].get(winner, [1.0]):
        candidate = build_pipeline(
            estimator(winner, config["seed"], c), config["features"], char=winner == "linear_svc"
        )
        candidate.fit(fit.review_text, fit.label)
        score = float(f1_score(val.label, candidate.predict(val.review_text), average="macro"))
        tuning.append({"C": c, "validation_macro_f1": score})
        if score > best_score:
            best, best_score = candidate, score
    LOG.info("Running validation-only preprocessing ablations")
    ablations = []
    for stop in (False, True):
        for ngrams in ((1, 1), (1, 2)):
            model = build_pipeline(
                estimator("logistic", config["seed"]),
                config["features"],
                stopwords=stop,
                ngrams=ngrams,
            )
            model.fit(fit.review_text, fit.label)
            ablations.append(
                {
                    "model": "logistic",
                    "stopwords": "negation_safe" if stop else "none",
                    "ngrams": str(ngrams),
                    "validation_macro_f1": float(
                        f1_score(val.label, model.predict(val.review_text), average="macro")
                    ),
                }
            )
    LOG.info("Calibrating frozen classifier on reserved training examples")
    calibrator = CalibratedClassifierCV(FrozenEstimator(best.named_steps["clf"]), method="sigmoid")
    calibrator.fit(best.named_steps["tfidf"].transform(calibration.review_text), calibration.label)
    best.set_params(clf=calibrator)
    optional = None
    if transformer:
        from src.transformer import train_transformer

        optional = train_transformer(fit, val, test, config, output / "transformer")
    # All selection and tuning is complete. Predict each test example exactly once.
    LOG.info("Final test evaluation")
    probabilities = best.predict_proba(test.review_text)
    predicted = best.classes_[probabilities.argmax(axis=1)]
    final = evaluate_predictions(test.label, predicted)
    domains = {
        source: evaluate_predictions(test.label[mask], predicted[mask])
        for source in sorted(test.source.unique())
        for mask in [test.source.to_numpy() == source]
    }
    metrics = {
        "schema_version": 2,
        "selected_model": winner,
        "test": final,
        "per_source": domains,
        "comparison": comparison,
        "validation_tuning": tuning,
        "ablations": ablations,
        "data_audit": audit,
        "seed": config["seed"],
        "max_samples": max_samples,
        "split_counts": {
            "train": len(train_df),
            "val": len(val),
            "test": len(test),
            "fit": len(fit),
            "calibration": len(calibration),
        },
        "split_class_distribution": {
            name: part.label.value_counts().to_dict()
            for name, part in [("train", train_df), ("val", val), ("test", test)]
        },
        "transformer": optional,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    evidence = test[["label", "source"]].rename(columns={"label": "true_label"}).copy()
    evidence["predicted_label"] = predicted
    add_credibility(metrics, evidence, config.get("evaluation"))
    final = metrics["test"]
    output.mkdir(parents=True, exist_ok=True)
    staging = output / "pipeline.joblib.tmp"
    joblib.dump(best, staging, compress=3)
    staging.replace(output / "pipeline.joblib")
    dataset_hash = hashlib.sha256(json.dumps(audit["sources"], sort_keys=True).encode()).hexdigest()
    metrics["dataset_hash"] = dataset_hash
    metrics["artifact_sha256"] = sha256_file(output / "pipeline.joblib")
    metadata = {
        "schema_version": 2,
        "selected_model": winner,
        "metrics": final,
        "dataset_hash": dataset_hash,
        "seed": config["seed"],
        "created_utc": metrics["created_utc"],
        "config": config,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("scikit-learn", "numpy", "scipy", "pandas", "joblib", "streamlit")
        },
        "artifact_sha256": metrics["artifact_sha256"],
        "calibration": "sigmoid on disjoint training calibration subset",
        "low_confidence_threshold": config["low_confidence_threshold"],
    }
    write_reports(reports, metrics, test, predicted, probabilities, best.classes_)
    (output / "metadata.json.tmp").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "metadata.json.tmp").replace(output / "metadata.json")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--transformer", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        config = load_config(args.config)
        with threadpool_limits(limits=1):
            train(
                config["paths"]["data"],
                config,
                max_samples=args.max_samples,
                transformer=args.transformer,
            )
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
