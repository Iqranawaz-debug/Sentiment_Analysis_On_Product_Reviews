"""Validation-only neutral experiments; test data is absent from selection APIs."""

import argparse
import copy
import hashlib
import itertools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.config import ROOT, load_config
from src.credibility import add_credibility
from src.data import LABELS, load_data, sha256_file, split_data
from src.decision import DecisionPolicy, RidgeOrdinal, predict_once
from src.evaluate import write_reports
from src.features import build_pipeline
from src.predict import load_artifact
from src.run_stats import RunStats
from src.structural_features import structured_pipeline

LOG = logging.getLogger(__name__)


def scores(labels, predicted):
    return {
        "validation_macro_f1": float(
            f1_score(labels, predicted, labels=LABELS, average="macro", zero_division=0)
        ),
        "validation_neutral_f1": float(
            f1_score(labels, predicted, labels=["neutral"], average="macro", zero_division=0)
        ),
    }


def fit_candidate(name, fit, calibration, config, parameter):
    if name == "ordinal_ridge":
        estimator = RidgeOrdinal(alpha=parameter)
        target = fit.star_rating
    else:
        estimator = LogisticRegression(
            C=parameter, class_weight="balanced", max_iter=2000, random_state=config["seed"]
        )
        target = fit.label
    builder = structured_pipeline if name == "structural_features" else build_pipeline
    pipeline = builder(estimator, config["features"])
    pipeline.fit(fit.review_text, target)
    base = pipeline.named_steps["clf"]
    calibrator = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrator.fit(
        pipeline.named_steps["tfidf"].transform(calibration.review_text), calibration.label
    )
    pipeline.set_params(clf=calibrator)
    if name == "ordinal_ridge":
        pipeline.set_params(
            clf=DecisionPolicy(calibrator, ordinal_estimator=base, thresholds=(2.5, 3.5))
        )
    return pipeline


def cv_summary(name, fit, config, parameter):
    """Descriptive fixed-setting CV; no validation-tuned weights/thresholds leak into CV."""
    cv = StratifiedKFold(config["cv_folds"], shuffle=True, random_state=config["seed"])
    results = []
    for train_index, held_index in cv.split(fit, fit.label):
        core, calibration = train_test_split(
            fit.iloc[train_index],
            test_size=config["calibration_fraction"],
            stratify=fit.iloc[train_index].label,
            random_state=config["seed"],
        )
        pipeline = fit_candidate(name, core, calibration, config, parameter)
        predicted, _ = predict_once(pipeline, fit.iloc[held_index].review_text)
        results.append(scores(fit.iloc[held_index].label, predicted))
    return {
        "cv_macro_f1_mean": float(np.mean([r["validation_macro_f1"] for r in results])),
        "cv_macro_f1_std": float(np.std([r["validation_macro_f1"] for r in results])),
        "cv_neutral_f1_mean": float(np.mean([r["validation_neutral_f1"] for r in results])),
        "cv_neutral_f1_std": float(np.std([r["validation_neutral_f1"] for r in results])),
        "cv_note": "Fixed settings; nested training-only calibration; untuned default decision rule",
    }


def select_experiment(reference, fit, calibration, validation, config):
    """The test frame cannot be passed here. Exact validation ties retain the reference."""
    cfg = config["experiments"]
    predictions, probabilities = predict_once(reference, validation.review_text)
    rows = [
        {
            "experiment": "reference_calibrated_logistic",
            **scores(validation.label, predictions),
            "parameters": "unchanged saved artifact",
        }
    ]
    models = [reference]
    trace = []
    # The original C is read from the frozen, fitted reference classifier.
    reference_c = reference.named_steps["clf"].calibrated_classifiers_[0].estimator.estimator.C
    if cfg["cv"]:
        rows[0].update(cv_summary("reference", fit, config, reference_c))
    weighted_best, weights_best, value_best = (
        reference,
        [1.0, 1.0, 1.0],
        rows[0]["validation_macro_f1"],
    )
    weighted_score = scores(validation.label, predictions)
    for negative, neutral in itertools.product(
        cfg["probability_weights"]["negative"], cfg["probability_weights"]["neutral"]
    ):
        weights = [negative, neutral, 1.0]
        predicted = reference.classes_[(probabilities * weights).argmax(axis=1)]
        result = scores(validation.label, predicted)
        trace.append(
            {"experiment": "probability_weighting", "parameters": {"weights": weights}, **result}
        )
        if result["validation_macro_f1"] > value_best:
            value_best, weights_best, weighted_score = (
                result["validation_macro_f1"],
                weights,
                result,
            )
            weighted_best = Pipeline(
                [
                    ("tfidf", reference.named_steps["tfidf"]),
                    ("clf", DecisionPolicy(reference.named_steps["clf"], weights=weights)),
                ]
            )
    rows.append(
        {
            "experiment": "probability_weighting",
            **weighted_score,
            "parameters": json.dumps({"weights": weights_best}),
            "cv_note": "N/A: post-fit policy tuned on validation; reference CV is not its CV estimate",
        }
    )
    models.append(weighted_best)
    for name, parameters in [
        ("ordinal_ridge", cfg["ordinal"]["alpha"]),
        ("structural_features", cfg["structural"]["C"]),
    ]:
        LOG.info("Validation experiment: %s", name)
        best_model, best_result, best_parameters = None, None, None
        for parameter in parameters:
            candidate = fit_candidate(name, fit, calibration, config, parameter)
            if name == "ordinal_ridge":
                features = candidate.named_steps["tfidf"].transform(validation.review_text)
                predicted_stars = candidate.named_steps["clf"].ordinal_estimator.predict_stars(
                    features
                )
                for lower, upper in itertools.product(
                    cfg["ordinal"]["lower_threshold"], cfg["ordinal"]["upper_threshold"]
                ):
                    if lower >= upper:
                        continue
                    predicted = candidate.classes_[np.digitize(predicted_stars, [lower, upper])]
                    result = scores(validation.label, predicted)
                    params = {"alpha": parameter, "lower": lower, "upper": upper}
                    trace.append({"experiment": name, "parameters": params, **result})
                    if (
                        best_result is None
                        or result["validation_macro_f1"] > best_result["validation_macro_f1"]
                    ):
                        best_result, best_parameters = result, params
                        best_model = Pipeline(
                            [
                                ("tfidf", candidate.named_steps["tfidf"]),
                                (
                                    "clf",
                                    DecisionPolicy(
                                        candidate.named_steps["clf"].calibrated_estimator,
                                        ordinal_estimator=candidate.named_steps[
                                            "clf"
                                        ].ordinal_estimator,
                                        thresholds=(lower, upper),
                                    ),
                                ),
                            ]
                        )
            else:
                predicted, _ = predict_once(candidate, validation.review_text)
                result = scores(validation.label, predicted)
                params = {"C": parameter}
                trace.append({"experiment": name, "parameters": params, **result})
                if (
                    best_result is None
                    or result["validation_macro_f1"] > best_result["validation_macro_f1"]
                ):
                    best_model, best_result, best_parameters = candidate, result, params
        row = {"experiment": name, **best_result, "parameters": json.dumps(best_parameters)}
        if cfg["cv"]:
            row.update(cv_summary(name, fit, config, cfg["cv_parameters"][name]))
        rows.append(row)
        models.append(best_model)
    winner = max(range(len(rows)), key=lambda i: rows[i]["validation_macro_f1"])
    for i, row in enumerate(rows):
        row["selected"] = i == winner
        for metric in (
            "cv_macro_f1_mean",
            "cv_macro_f1_std",
            "cv_neutral_f1_mean",
            "cv_neutral_f1_std",
        ):
            row.setdefault(metric, None)
    return models[winner], rows, trace


def run_experiments(config, max_samples, allow_full=False):
    if max_samples is None and not allow_full:
        raise ValueError(
            "Full-corpus training requires confirmation; use --allow-full only after approval."
        )
    stats = RunStats()
    output, reports = Path(config["paths"]["models"]), Path(config["paths"]["reports"])
    if (reports / "experiments.csv").exists():
        raise ValueError(
            "Experiments already reported. Choose fresh output directories; do not retune on test."
        )
    reference_models = ROOT / config["experiments"]["reference_models"]
    reference_reports = ROOT / config["experiments"]["reference_reports"]
    reference, reference_meta = load_artifact(reference_models)
    previous = json.loads((reference_reports / "metrics.json").read_text(encoding="utf-8"))
    if previous["artifact_sha256"] != reference_meta["artifact_sha256"]:
        raise ValueError("Reference artifact and report mismatch.")
    if previous["max_samples"] != max_samples or previous["seed"] != config["seed"]:
        raise ValueError(
            "Reference must use the identical sample cap and seed; train a matching reference first."
        )
    frame, audit = load_data(config["paths"]["data"])
    dataset_hash = hashlib.sha256(json.dumps(audit["sources"], sort_keys=True).encode()).hexdigest()
    if dataset_hash != previous["dataset_hash"]:
        raise ValueError("Dataset differs from the reference; its split cannot be reused safely.")
    train, validation, test = split_data(frame, config["seed"], max_samples)
    fit, calibration = train_test_split(
        train,
        test_size=config["calibration_fraction"],
        stratify=train.label,
        random_state=config["seed"],
    )
    best, experiments, trace = select_experiment(reference, fit, calibration, validation, config)
    selected = next(row for row in experiments if row["selected"])
    reports.mkdir(parents=True, exist_ok=True)
    # Seal the decision before any final test access. Crash recovery must not rerun selection against test.
    pd.DataFrame(experiments).to_csv(reports / "experiments.csv", index=False)
    (reports / "validation_search.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    LOG.info("Validation winner locked: %s", selected["experiment"])
    if selected["experiment"] == "reference_calibrated_logistic":
        cached = pd.read_csv(reference_reports / "predictions.csv")
        if (
            cached.review_text.tolist() != test.review_text.tolist()
            or cached.true_label.tolist() != test.label.tolist()
        ):
            raise ValueError("Cached reference test rows do not match the reconstructed split.")
        predictions = cached.predicted_label.to_numpy()
        probabilities = cached[[f"probability_{label}" for label in best.classes_]].to_numpy()
        test_pass = "Reused original saved reference predictions; no additional model test pass"
    else:
        LOG.info("One final test prediction pass for the validation-selected configuration")
        predictions, probabilities = predict_once(best, test.review_text)
        test_pass = "One final prediction pass for the selected configuration only"
    metrics = copy.deepcopy(previous)
    metrics.update(
        selected_model=selected["experiment"],
        experiments=experiments,
        experiments_config=config["experiments"],
        created_utc=datetime.now(timezone.utc).isoformat(),
        protocol="Follow-up on the previously reported fixed holdout; all new selection is validation-only",
        test_evaluation=test_pass,
        reference_artifact_sha256=reference_meta["artifact_sha256"],
        beats_reference=selected["validation_macro_f1"] > experiments[0]["validation_macro_f1"],
    )
    rows = test[["label", "source"]].rename(columns={"label": "true_label"}).copy()
    rows["predicted_label"] = predictions
    add_credibility(metrics, rows, config.get("evaluation"))
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, output / "pipeline.joblib.tmp", compress=3)
    (output / "pipeline.joblib.tmp").replace(output / "pipeline.joblib")
    metrics["artifact_sha256"] = sha256_file(output / "pipeline.joblib")
    write_reports(reports, metrics, test, predictions, probabilities, best.classes_)
    metadata = copy.deepcopy(reference_meta)
    metadata.update(
        selected_model=selected["experiment"],
        metrics=metrics["test"],
        config=config,
        created_utc=metrics["created_utc"],
        artifact_sha256=metrics["artifact_sha256"],
        decision_policy=selected["parameters"],
        reference_artifact_sha256=reference_meta["artifact_sha256"],
        max_samples=max_samples,
        **stats.snapshot(),
    )
    metadata["timing_scope"] = (
        "Follow-up data loading, experiments, CV and reports; excludes prior reference training"
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    LOG.info(
        "Finished in %.1f seconds; peak process RSS %.1f MiB",
        metadata["training_seconds"],
        metadata["peak_memory_mb"],
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/neutral.yaml")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--allow-full",
        action="store_true",
        help="Only after user approval for full-corpus training",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        with threadpool_limits(limits=1):
            run_experiments(load_config(args.config), args.max_samples, args.allow_full)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Experiment failed: {error}\n")


if __name__ == "__main__":
    main()
