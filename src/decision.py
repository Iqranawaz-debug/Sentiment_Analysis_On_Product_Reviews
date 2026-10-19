"""Serializable decision policies; calibrated probabilities are never reweighted."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Ridge
from sklearn.utils.class_weight import compute_sample_weight

from src.data import LABELS


class RidgeOrdinal(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        stars = np.asarray(y, dtype=float)
        if not np.isin(stars, [1, 2, 3, 4, 5]).all():
            raise ValueError(
                "Ordinal training requires original star ratings; labels cannot reconstruct them."
            )
        labels = np.where(stars < 3, "negative", np.where(stars > 3, "positive", "neutral"))
        self.regressor_ = Ridge(alpha=self.alpha, solver="lsqr").fit(
            X, stars, sample_weight=compute_sample_weight("balanced", labels)
        )
        self.classes_ = np.array(LABELS)
        return self

    def predict_stars(self, X):
        return self.regressor_.predict(X)

    def decision_function(self, X):
        return -np.abs(self.predict_stars(X)[:, None] - np.array([1.5, 3.0, 4.5]))

    def predict(self, X):
        return self.classes_[np.digitize(self.predict_stars(X), [2.5, 3.5])]


class DecisionPolicy(ClassifierMixin, BaseEstimator):
    def __init__(self, calibrated_estimator, weights=None, ordinal_estimator=None, thresholds=None):
        self.calibrated_estimator = calibrated_estimator
        self.weights = weights
        self.ordinal_estimator = ordinal_estimator
        self.thresholds = thresholds

    def fit(self, X, y=None):
        return self

    @property
    def classes_(self):
        return self.calibrated_estimator.classes_

    def __sklearn_is_fitted__(self):
        return hasattr(self.calibrated_estimator, "calibrated_classifiers_")

    def predict_proba(self, X):
        return self.calibrated_estimator.predict_proba(X)

    def labels_from_probabilities(self, probabilities, X):
        if self.ordinal_estimator is not None:
            return self.classes_[
                np.digitize(self.ordinal_estimator.predict_stars(X), self.thresholds)
            ]
        weights = np.ones(len(self.classes_)) if self.weights is None else np.asarray(self.weights)
        return self.classes_[(probabilities * weights).argmax(axis=1)]

    def predict(self, X):
        return self.labels_from_probabilities(self.predict_proba(X), X)


def predict_once(pipeline, texts):
    """One final reporting call; all subsequent analysis uses these cached outputs."""
    features = pipeline.named_steps["tfidf"].transform(texts)
    classifier = pipeline.named_steps["clf"]
    probabilities = classifier.predict_proba(features)
    if isinstance(classifier, DecisionPolicy):
        labels = classifier.labels_from_probabilities(probabilities, features)
    else:
        labels = classifier.classes_[probabilities.argmax(axis=1)]
    return labels, probabilities
