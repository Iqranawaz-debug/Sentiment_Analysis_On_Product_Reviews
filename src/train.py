"""CLI training: python -m src.train --data first.csv second.csv."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import time

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data import load_data, sha256_file
from src.evaluation import evaluate
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def candidates(seed, include_random_forest=False):
    models = {
        "Dummy baseline": (DummyClassifier(strategy="most_frequent"), {}),
        "Naive Bayes": (MultinomialNB(), {"clf__alpha": [0.1, 1.0]}),
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=seed),
                                {"clf__C": [0.5, 2.0], "clf__class_weight": [None, "balanced"]}),
        "Linear SVM": (LinearSVC(random_state=seed, max_iter=5000),
                       {"clf__C": [0.5, 2.0], "clf__class_weight": [None, "balanced"]}),
    }
    if include_random_forest:
        models["Random Forest"] = (RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1),
                                   {"clf__max_depth": [30, None], "clf__class_weight": ["balanced"]})
    return models


def build_pipeline(classifier, config):
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, lowercase=False,
                                 ngram_range=(1, 2), min_df=config["min_df"],
                                 max_features=config["max_features"], sublinear_tf=True)),
        ("clf", classifier),
    ])


def train(paths, config, output):
    if not 0 < config["test_size"] < 1 or config["cv_folds"] < 2:
        raise ValueError("test_size must be between 0 and 1, and cv_folds at least 2.")
    if config["sample_size"] < 0 or config["min_df"] < 1 or config["max_features"] < 1:
        raise ValueError("sample_size must be non-negative; min_df and max_features must be positive.")
    seed = config["seed"]
    frame, audit = load_data(paths)
    sample_size = config["sample_size"]
    if sample_size and sample_size < len(frame):
        frame, _ = train_test_split(frame, train_size=sample_size, stratify=frame.label, random_state=seed)
    train_df, test_df = train_test_split(frame, test_size=config["test_size"], stratify=frame.label, random_state=seed)
    if train_df.label.value_counts().min() < config["cv_folds"]:
        raise ValueError("Too few examples in a training class for the requested CV folds.")
    assert not set(train_df.normalized) & set(test_df.normalized)
    cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=seed)
    comparison, searches = [], {}
    for name, (classifier, grid) in candidates(seed, config["include_random_forest"]).items():
        LOG.info("Comparing %s", name)
        start = time.perf_counter()
        search = GridSearchCV(build_pipeline(classifier, config), grid, scoring={"macro_f1": "f1_macro", "accuracy": "accuracy", "weighted_f1": "f1_weighted"},
                              refit="macro_f1", cv=cv, n_jobs=config["n_jobs"], error_score="raise")
        search.fit(train_df.review_text, train_df.label)
        best = search.best_index_
        comparison.append({"model": name, "cv_macro_f1": float(search.best_score_),
                           "cv_macro_f1_std": float(search.cv_results_["std_test_macro_f1"][best]),
                           "cv_accuracy": float(search.cv_results_["mean_test_accuracy"][best]),
                           "cv_weighted_f1": float(search.cv_results_["mean_test_weighted_f1"][best]),
                           "best_parameters": search.best_params_, "search_seconds": time.perf_counter() - start})
        searches[name] = search
    winner = max(comparison, key=lambda row: row["cv_macro_f1"])["model"]
    pipeline = searches[winner].best_estimator_
    # Holdout is accessed only after selection. Do not select again from these metrics.
    metrics = evaluate(pipeline, test_df.review_text, test_df.label)
    metadata = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": winner, "selection_rule": "Highest mean training CV macro F1; insertion order breaks exact ties.",
        "config": config, "versions": {p: importlib.metadata.version(p) for p in ["scikit-learn", "numpy", "scipy", "pandas", "joblib", "streamlit"]},
        "python": platform.python_version(), "data_audit": audit,
        "sample_rows": len(frame), "train_rows": len(train_df), "test_rows": len(test_df),
        "split_class_distribution": {name: part.label.value_counts().sort_index().to_dict() for name, part in [("train", train_df), ("test", test_df)]},
        "normalized_overlap": 0, "comparison": comparison, "holdout": metrics,
        "training_length_quantiles": train_df.review_text.str.len().quantile([0, .25, .5, .75, .95, 1]).to_dict(),
        "limitations": ["Rating-derived labels are sentiment proxies.", "Random review split does not isolate products, authors, or time.", "Near duplicates may remain.", "English-focused features; probabilities are not calibrated.", "Holdout represents the deduplicated sample, not live traffic."],
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Write model first; checksum makes a partially replaced pair fail closed.
    staging = output / "pipeline.joblib.tmp"
    joblib.dump(pipeline, staging, compress=3)
    staging.replace(output / "pipeline.joblib")
    metadata["artifact_sha256"] = sha256_file(output / "pipeline.joblib")
    metadata["artifact_bytes"] = (output / "pipeline.joblib").stat().st_size
    (output / "metadata.json.tmp").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    (output / "metadata.json.tmp").replace(output / "metadata.json")
    LOG.info("Selected %s; holdout macro F1 %.4f", winner, metrics["macro_f1"])
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/training.json")
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        train(args.data, json.loads(args.config.read_text(encoding="utf-8")), args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
