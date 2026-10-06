"""Metrics with explicit class ordering."""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data import LABELS


def evaluate(model, texts, labels):
    predictions = model.predict(texts)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "classification_report": classification_report(labels, predictions, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=LABELS).tolist(),
        "class_order": LABELS,
    }
