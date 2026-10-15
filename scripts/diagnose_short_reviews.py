"""Read-only model diagnosis and hand-authored sanity evaluation. Never trains."""

import json
from pathlib import Path

import numpy as np

from sentix.predictor import load_model
from src.data import sha256_file
from src.explain import linear_estimator

CASES = {
    "positive": [
        "I love it",
        "Absolutely love it",
        "Excellent product",
        "Works perfectly",
        "Great quality",
        "Very happy with this",
        "Highly recommend",
        "Not bad at all",
    ],
    "negative": [
        "Absolutely hate it",
        "I hate it",
        "Terrible product",
        "It broke immediately",
        "Waste of money",
        "Not good",
        "Does not work",
        "I do not recommend it",
    ],
    "neutral": [
        "It is average",
        "It is okay",
        "Nothing special",
        "Neither good nor bad",
        "An ordinary product",
        "It is just acceptable",
        "Average quality",
        "It is mediocre",
    ],
}


def main() -> None:
    predictor = load_model()
    model = predictor.model
    estimator = linear_estimator(model)
    vectorizer = model.named_steps["tfidf"]
    text = "Absolutely hate it"
    features = vectorizer.transform([text]).tocoo()
    names = vectorizer.get_feature_names_out()
    classes = list(estimator.classes_)
    terms = []
    for index, value in zip(features.col, features.data):
        terms.append(
            {
                "term": names[index],
                "tfidf": float(value),
                "coefficients": dict(zip(classes, estimator.coef_[:, index].tolist())),
                "contributions": dict(zip(classes, (estimator.coef_[:, index] * value).tolist())),
            }
        )
    analyzer_terms = vectorizer.build_analyzer()(text)
    rows = []
    for expected, reviews in CASES.items():
        for review in reviews:
            result = predictor.predict(review)
            rows.append(
                {
                    "text": review,
                    "expected": expected,
                    **result,
                    "correct": result["label"] == expected,
                }
            )
    report = {
        "artifact_sha256": sha256_file(Path("models/pipeline.joblib")),
        "scope": "24 hand-authored English sanity cases, 8 per class; not an independent benchmark. Labels are author judgments; short neutral phrases can be ambiguous.",
        "diagnosis": {
            "text": text,
            "prediction": predictor.predict(text),
            "features": terms,
            "absent_features": [
                term for term in analyzer_terms if term not in vectorizer.vocabulary_
            ],
            "intercepts": dict(zip(classes, estimator.intercept_.tolist())),
            "raw_scores": dict(
                zip(classes, estimator.decision_function(vectorizer.transform([text]))[0].tolist())
            ),
            "ngram_range": vectorizer.ngram_range,
            "class_weight": estimator.class_weight,
            "calibration": predictor.metadata.get("calibration"),
        },
        "accuracy": float(np.mean([row["correct"] for row in rows])),
        "correct": sum(row["correct"] for row in rows),
        "total": len(rows),
        "reviews": rows,
    }
    destination = Path("reports/short_review_sanity.json")
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
