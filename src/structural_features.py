"""Text-only structural features for the optional ColumnTransformer experiment."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from src.preprocess import clean_text


class TextFrame(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame({"review_text": list(X)})

    def get_feature_names_out(self, input_features=None):
        return np.array(["review_text"])


class ReviewStructure(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            words = clean_text(text).split()
            length = len(words)
            rows.append(
                [
                    length < 10,
                    10 <= length < 50,
                    50 <= length < 150,
                    length >= 150,
                    text.count("!"),
                    text.count("?"),
                    sum(word in {"not", "no", "nor", "never"} for word in words),
                ]
            )
        return sparse.csr_matrix(np.array(rows, dtype=float))

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [
                "length_under_10",
                "length_10_49",
                "length_50_149",
                "length_150_plus",
                "exclamation_count",
                "question_count",
                "negation_count",
            ]
        )


def structured_pipeline(classifier, config):
    text = TfidfVectorizer(
        preprocessor=clean_text,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=config["min_df"],
        max_features=config["max_features"],
        sublinear_tf=True,
    )
    columns = ColumnTransformer(
        [
            ("text", text, "review_text"),
            (
                "structure",
                Pipeline([("counts", ReviewStructure()), ("scale", MaxAbsScaler())]),
                "review_text",
            ),
        ],
        sparse_threshold=1.0,
    )
    features = Pipeline([("frame", TextFrame()), ("columns", columns)])
    return Pipeline([("tfidf", features), ("clf", classifier)])
