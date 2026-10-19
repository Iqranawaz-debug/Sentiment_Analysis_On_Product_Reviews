"""Generate reports from a single test prediction pass; CLI only reads saved evidence."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data import LABELS


def evaluate_predictions(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=LABELS, average="macro", zero_division=0)
        ),
        "per_class": classification_report(
            labels, predictions, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=LABELS).tolist(),
        "class_order": LABELS,
        "support": len(labels),
    }


def write_reports(directory, metrics, test, predicted, probabilities, classes):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    comparison = list(metrics["comparison"])
    if metrics.get("transformer"):
        optional = metrics["transformer"]
        comparison.append(
            {
                "model": optional["model"],
                "cv_macro_f1": None,
                "cv_macro_f1_std": None,
                "test_accuracy": optional["test"]["accuracy"],
                "test_macro_f1": optional["test"]["macro_f1"],
                "protocol": optional["note"],
            }
        )
        (directory / "transformer_metrics.json").write_text(
            json.dumps(optional, indent=2, allow_nan=False), encoding="utf-8"
        )
    pd.DataFrame(comparison).to_csv(directory / "comparison.csv", index=False)
    pd.DataFrame(metrics["ablations"]).to_csv(directory / "ablation.csv", index=False)
    rows = test[["review_text", "label", "source"]].rename(columns={"label": "true_label"}).copy()
    if "star_rating" in test:
        rows["star_rating"] = test.star_rating.to_numpy()
    rows["predicted_label"] = predicted
    class_indices = np.array([list(classes).index(label) for label in predicted])
    rows["confidence"] = probabilities[np.arange(len(rows)), class_indices]
    for index, label in enumerate(classes):
        rows[f"probability_{label}"] = probabilities[:, index]
    rows.to_csv(directory / "predictions.csv", index=False)
    from src.error_analysis import neutral_error_sample

    neutral_error_sample(rows, metrics["seed"]).to_csv(
        directory / "neutral_error_analysis.csv", index=False
    )
    rows.loc[rows.true_label != rows.predicted_label].sort_values(
        "confidence", ascending=False
    ).head(50).to_csv(directory / "error_analysis.csv", index=False)
    for name, result in {"test": metrics["test"], **metrics["per_source"]}.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        matrix = np.array(result["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        ax.set(
            xticks=range(len(LABELS)),
            yticks=range(len(LABELS)),
            xticklabels=LABELS,
            yticklabels=LABELS,
            xlabel="Predicted",
            ylabel="True",
            title=f"Confusion matrix: {name}",
        )
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
            )
        fig.tight_layout()
        fig.savefig(figures / f"confusion_{name}.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    curve_rows = []
    for index, label in enumerate(classes):
        truth, prob = calibration_curve(test.label == label, probabilities[:, index], n_bins=10)
        ax.plot(prob, truth, marker="o", label=label)
        curve_rows.extend(
            {"class": label, "mean_probability": float(p), "fraction_positive": float(t)}
            for p, t in zip(prob, truth)
        )
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="Test reliability: one-vs-rest",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "calibration.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(curve_rows).to_csv(directory / "calibration.csv", index=False)
    (directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    (directory / "results.md").write_text(results_markdown(metrics), encoding="utf-8")


def results_markdown(metrics):
    lines = [
        f"Selected model: **{metrics['selected_model']}**.",
        "",
        "| Test population | Macro-F1 (95% CI) | Majority macro-F1 | Accuracy (95% CI) | Majority accuracy | Support |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, result in {"All": metrics["test"], **metrics["per_source"]}.items():

        def interval(key):
            ci = result.get("confidence_intervals", {}).get(key)
            value = f"{result[key]:.4f}"
            return value + (f" [{ci['lower']:.4f}, {ci['upper']:.4f}]" if ci else "")

        baseline = result.get("majority_baseline", {})
        baseline_f1 = f"{baseline['macro_f1']:.4f}" if baseline else "pending"
        baseline_acc = f"{baseline['accuracy']:.4f}" if baseline else "pending"
        support = str(result["support"]) + (
            "; **low support**" if result.get("low_support") else ""
        )
        lines.append(
            f"| {name} | {interval('macro_f1')} | {baseline_f1} | {interval('accuracy')} | {baseline_acc} | {support} |"
        )
    lines += ["", "| Model | Training CV macro-F1 | Fold SD |", "|---|---:|---:|"]
    lines += [
        f"| {r['model']} | {r['cv_macro_f1']:.4f} | {r['cv_macro_f1_std']:.4f} |"
        for r in metrics["comparison"]
    ]
    lines += ["", "| Class | Precision | Recall | F1 |", "|---|---:|---:|---:|"]
    lines += [
        f"| {label} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} |"
        for label in LABELS
        for r in [metrics["test"]["per_class"][label]]
    ]
    lines += [
        "",
        f"Run sample cap: {metrics['max_samples'] or 'none (full dataset)'}. Split counts: `{metrics['split_counts']}`.",
        "CV uses the fitting portion of training; calibration examples are excluded from selection.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--update-readme", type=Path)
    args = parser.parse_args()
    metrics = json.loads((args.reports / "metrics.json").read_text(encoding="utf-8"))
    result = results_markdown(metrics)
    (args.reports / "results.md").write_text(result, encoding="utf-8")
    if args.update_readme:
        content = args.update_readme.read_text(encoding="utf-8")
        start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        before, tail = content.split(start)
        _, after = tail.split(end)
        args.update_readme.write_text(
            before + start + "\n" + result + end + after, encoding="utf-8"
        )
    print(result)


if __name__ == "__main__":
    main()
