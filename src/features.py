"""Sparse CPU-friendly word and character features."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from src.preprocess import clean_text, without_stopwords


def build_pipeline(classifier, config, char=False, stopwords=False, ngrams=(1, 2)):
    normalize = without_stopwords if stopwords else clean_text
    word = TfidfVectorizer(preprocessor=normalize, lowercase=False, ngram_range=ngrams,
                           min_df=config["min_df"], max_features=config["max_features"],
                           sublinear_tf=True)
    features = word
    if char:
        features = FeatureUnion([
            ("word", word),
            ("char", TfidfVectorizer(preprocessor=normalize, lowercase=False,
                                     analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=config["min_df"], sublinear_tf=True,
                                     max_features=config.get("char_max_features", 50000)))])
    return Pipeline([("tfidf", features), ("clf", classifier)])
