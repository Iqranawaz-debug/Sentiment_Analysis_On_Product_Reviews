"""Isolated, staged experiments. Production files are read-only inputs."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from threadpoolctl import threadpool_limits

from scripts.diagnose_short_reviews import CASES
from src.data import LABELS, load_data, sha256_file, split_data
from src.features import build_pipeline
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports/approved_20261016"
MODELS = ROOT / "models/candidates/approved_20261016"
SEED = 20261016


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def production_hashes():
    return {p.name: sha256_file(p) for p in (ROOT / "models").iterdir() if p.is_file()}


def guard():
    assert production_hashes() == json.loads((RUN / "production_hashes.json").read_text())


def prepare():
    if RUN.exists():
        raise RuntimeError("Fresh run directory required; do not regenerate inspected splits.")
    RUN.mkdir(parents=True)
    write(RUN / "production_hashes.json", production_hashes())
    meta = json.loads((ROOT / "reports/metrics.json").read_text())
    frame, audit = load_data(
        [ROOT / "data/raw" / s["filename"] for s in meta["data_audit"]["sources"]]
    )
    assert (
        hashlib.sha256(json.dumps(audit["sources"], sort_keys=True).encode()).hexdigest()
        == meta["dataset_hash"]
    )
    old = pd.concat(split_data(frame, meta["seed"], meta["max_samples"]), ignore_index=True)
    frame = frame[~frame.normalized.isin(old.normalized)].copy()
    regression = {clean_text(t) for cases in CASES.values() for t in cases}
    frame = frame[~frame.normalized.isin(regression)].copy()
    frame["words"] = frame.normalized.str.split().str.len()
    base, _ = train_test_split(frame, train_size=16000, stratify=frame.label, random_state=SEED)
    extra = frame[(frame.words <= 8) & ~frame.normalized.isin(base.normalized)]
    extra = extra.sample(min(2000, len(extra)), random_state=SEED)
    base = base.assign(extra=False)
    extra = extra.assign(extra=True)
    pool = pd.concat([base, extra], ignore_index=True)
    # Fixed, label-blind duplicate screening; vectorizer is used only for split hygiene.
    all_text = pd.concat([old.normalized, pool.normalized], ignore_index=True)
    vectors = HashingVectorizer(
        n_features=2**20, alternate_sign=False, ngram_range=(1, 2)
    ).transform(all_text)
    nold = len(old)
    parent = list(range(len(pool)))
    rejected = set()

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(pool), 64):
        sims = (vectors[nold + start : nold + min(start + 64, len(pool))] @ vectors.T).tocoo()
        for r, c in zip(sims.row[sims.data >= 0.92], sims.col[sims.data >= 0.92]):
            i = start + int(r)
            if c < nold:
                rejected.add(i)
            elif c - nold != i:
                parent[find(i)] = find(int(c) - nold)
        if start % 1024 == 0:
            print("duplicate screen", start, "/", len(pool), flush=True)
    bad_groups = {find(i) for i in rejected}
    pool["group"] = [find(i) for i in range(len(pool))]
    pool = pool[~pool.group.isin(bad_groups)].copy().reset_index(drop=True)
    # Also keep simple lexical templates together (numbers and word order variations).
    # Cosine screen above captures the overwhelming majority of such near matches.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    a, b = next(splitter.split(pool, groups=pool.group))
    train, rest = pool.iloc[a].copy(), pool.iloc[b].copy()
    a, b = next(
        GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED).split(
            rest, groups=rest.group
        )
    )
    val, test = rest.iloc[a].copy(), rest.iloc[b].copy()
    a, b = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(
            train, groups=train.group
        )
    )
    parts = {"fit": train.iloc[a], "calibration": train.iloc[b], "validation": val, "test": test}
    for name, part in parts.items():
        part.to_csv(RUN / f"{name}.csv", index=False)
    for x, part in parts.items():
        for y, other in parts.items():
            if x != y:
                assert not set(part.group) & set(other.group)
    short = parts["fit"][parts["fit"].words <= 8]
    audit_sample = short.groupby("label", group_keys=False).sample(n=12, random_state=SEED)
    audit_sample[["review_text", "label", "star_rating", "normalized"]].to_csv(
        RUN / "training_label_audit.csv", index=False
    )
    write(
        RUN / "protocol.json",
        {
            "seed": SEED,
            "production_excluded_rows": len(old),
            "source_audit": audit,
            "near_duplicate_method": "nonnegative hashed unigram/bigram cosine >=0.92; connected components; no exhaustive semantic paraphrase guarantee",
            "near_production_removed": len(base) + len(extra) - len(pool),
            "extra_short_sample": len(extra),
            "test_distribution": "short-review enriched, rating-derived labels; not deployment prevalence",
            "split_counts": {k: len(v) for k, v in parts.items()},
            "split_sha256": {k: sha256_file(RUN / f"{k}.csv") for k in parts},
            "selection": "maximum validation macro-F1, ties first; never regression or test",
            "threshold_grid": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            "threshold_rule": "lowest validation threshold attaining >=90% accuracy on accepted rows; otherwise no threshold",
            "transformer": "cardiffnlp/twitter-roberta-base-sentiment-latest, frozen, no task fine-tuning",
            "hybrid_threshold": 0.80,
        },
    )
    guard()
    print("Prepared", {k: len(v) for k, v in parts.items()}, flush=True)


def metrics(frame, probs):
    pred = np.array(LABELS)[probs.argmax(axis=1)]
    return {
        "accuracy": accuracy_score(frame.label, pred),
        "macro_f1": f1_score(frame.label, pred, average="macro"),
        "per_class": classification_report(
            frame.label, pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "class_order": LABELS,
        "confusion_matrix": confusion_matrix(frame.label, pred, labels=LABELS).tolist(),
        "n": len(frame),
        "short": slice_metric(frame, pred, frame.words <= 8),
        "long": slice_metric(frame, pred, frame.words > 8),
    }


def slice_metric(frame, pred, mask):
    return {
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(frame.label[mask], pred[mask])) if mask.any() else None,
    }


def probabilities(model, texts):
    p = model.predict_proba(list(texts))
    return p[:, [list(model.classes_).index(c) for c in LABELS]]


def train():
    guard()
    if (RUN / "selection.json").exists():
        raise RuntimeError("Selection already frozen")
    fit, cal, val = [pd.read_csv(RUN / f"{n}.csv") for n in ["fit", "calibration", "validation"]]
    # Apply only explicit, training-only audit decisions; retain proxy labels elsewhere.
    audit_path = RUN / "audit_decisions.json"
    if audit_path.exists():
        decisions = json.loads(audit_path.read_text())
        for text, label in decisions["corrections"].items():
            fit.loc[fit.normalized == text, "label"] = label
        fit = fit[~fit.normalized.isin(decisions["exclude_ambiguous"])].copy()
    specs = [
        {
            "name": "baseline_fresh",
            "extra": False,
            "min_df": 2,
            "max_features": 50000,
            "ngrams": [1, 2],
            "C": 2.0,
        },
        {
            "name": "extra_short",
            "extra": True,
            "min_df": 2,
            "max_features": 50000,
            "ngrams": [1, 2],
            "C": 2.0,
        },
        {
            "name": "coverage_c1",
            "extra": True,
            "min_df": 1,
            "max_features": 80000,
            "ngrams": [1, 2],
            "C": 1.0,
        },
        {
            "name": "trigrams_c2",
            "extra": True,
            "min_df": 1,
            "max_features": 80000,
            "ngrams": [1, 3],
            "C": 2.0,
        },
    ]
    rows = []
    for spec in specs:
        rows.append(fit_one(spec, fit, cal, val))
    winner = max(rows, key=lambda r: r["validation"]["macro_f1"])
    chosen = copy.deepcopy(winner["config"])
    chosen["name"] = "short_calibration"
    chosen["short_calibration"] = True
    rows.append(fit_one(chosen, fit, cal, val))
    winner = max(rows, key=lambda r: r["validation"]["macro_f1"])
    model = joblib.load(MODELS / winner["name"] / "pipeline.joblib")
    p = probabilities(model, val.review_text)
    pred = np.array(LABELS)[p.argmax(axis=1)]
    coverage = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        mask = p.max(axis=1) >= threshold
        coverage.append(
            {
                "threshold": threshold,
                "coverage": float(mask.mean()),
                "accuracy": float(accuracy_score(val.label[mask], pred[mask]))
                if mask.any()
                else None,
            }
        )
    accepted = [r for r in coverage if r["accuracy"] is not None and r["accuracy"] >= 0.90]
    write(
        RUN / "selection.json",
        {
            "candidates": rows,
            "winner": winner["name"],
            "abstention_validation": coverage,
            "abstention_threshold": accepted[0]["threshold"] if accepted else None,
        },
    )
    guard()


def fit_one(spec, fit, cal, val):
    out = MODELS / spec["name"]
    out.mkdir(parents=True, exist_ok=False)
    data = fit if spec["extra"] else fit[~fit.extra]
    calibration = cal if spec["extra"] else cal[~cal.extra]
    if spec.get("short_calibration"):
        calibration = calibration[calibration.words <= 8]
    assert set(calibration.label) == set(LABELS)
    model = build_pipeline(
        LogisticRegression(C=spec["C"], class_weight="balanced", max_iter=2000, random_state=SEED),
        spec,
        ngrams=tuple(spec["ngrams"]),
    )
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        model.fit(data.review_text, data.label)
        calibrator = CalibratedClassifierCV(
            FrozenEstimator(model.named_steps["clf"]), method="sigmoid"
        )
        calibrator.fit(
            model.named_steps["tfidf"].transform(calibration.review_text), calibration.label
        )
        model.set_params(clf=calibrator)
    elapsed = time.perf_counter() - start
    joblib.dump(model, out / "pipeline.joblib", compress=3)
    config = {
        **spec,
        "seed": SEED,
        "class_weight": "balanced",
        "fit_rows": len(data),
        "calibration_rows": len(calibration),
        "calibration": "sigmoid on disjoint reserved rows",
        "protocol_sha256": sha256_file(RUN / "protocol.json"),
    }
    row = {
        "name": spec["name"],
        "config": config,
        "train_seconds": elapsed,
        "sha256": sha256_file(out / "pipeline.joblib"),
        "size_bytes": (out / "pipeline.joblib").stat().st_size,
        "validation": metrics(val, probabilities(model, val.review_text)),
    }
    write(out / "training_config.json", row)
    print(spec["name"], row["validation"]["macro_f1"], flush=True)
    return row


def evaluate():
    guard()
    if (RUN / "comparison.json").exists():
        raise RuntimeError("Test already evaluated; no tuning after test")
    selection = json.loads((RUN / "selection.json").read_text())
    test = pd.read_csv(RUN / "test.csv")
    regression = pd.DataFrame(
        [
            {"review_text": t, "label": label, "words": len(clean_text(t).split())}
            for label, items in CASES.items()
            for t in items
        ]
    )
    rows = []
    candidates = [("production", ROOT / "models/pipeline.joblib")] + [
        (r["name"], MODELS / r["name"] / "pipeline.joblib") for r in selection["candidates"]
    ]
    for name, path in candidates:
        model = joblib.load(path)
        probabilities(model, test.review_text.iloc[:3])
        start = time.perf_counter()
        p = probabilities(model, test.review_text)
        elapsed = time.perf_counter() - start
        rp = probabilities(model, regression.review_text)
        row = {
            "name": name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "test": metrics(test, p),
            "regression": metrics(regression, rp),
            "test_seconds": elapsed,
            "amortized_ms_per_review": elapsed / len(test) * 1000,
        }
        save_predictions(name, test, p, regression, rp)
        rows.append(row)
    write(RUN / "comparison.json", rows)
    guard()


def save_predictions(name, test, p, regression, rp):
    for suffix, frame, probs in [("test", test, p), ("regression", regression, rp)]:
        result = frame[["review_text", "label", "words"]].copy()
        result["predicted"] = np.array(LABELS)[probs.argmax(axis=1)]
        for j, label in enumerate(LABELS):
            result[label] = probs[:, j]
        result.to_csv(RUN / f"{name}_{suffix}_predictions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "evaluate"])
    globals()[parser.parse_args().stage]()
