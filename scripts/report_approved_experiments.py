"""Render measured results; this module never selects or trains models."""

import json

import pandas as pd

from scripts.run_approved_experiments import LABELS, MODELS, RUN, guard, write


def main():
    guard()
    rows = json.loads((RUN / "comparison.json").read_text())
    transformer_path = RUN / "transformer_results.json"
    if transformer_path.exists():
        rows += json.loads(transformer_path.read_text())
    protocol = json.loads((RUN / "protocol.json").read_text())
    selection = json.loads((RUN / "selection.json").read_text())
    test = pd.read_csv(RUN / "test.csv")
    lines = [
        "# Approved sentiment experiments — 2026-10-16",
        "",
        "Run started 2026-10-16; final artifact and routing verification completed 2026-10-19.",
        "",
        "Production artifacts were read only. Every trained candidate is stored separately under `models/candidates/approved_20261016/`.",
        "",
        "## Protocol and limitations",
        "",
        f"Excluded all {protocol['production_excluded_rows']:,} rows from the production sample (training, calibration, validation, and old test). Source checksums matched the original report.",
        f"New split sizes: {protocol['split_counts']}. Removed {protocol['near_production_removed']} sampled rows in near-duplicate components connected to production.",
        f"Duplicate screen: {protocol['near_duplicate_method']}. Connected components stay in one split. This does not establish product/user-level or semantic independence.",
        "The 24 exposed cases were excluded by normalized text, never used for fitting, calibration, hyperparameter selection, or threshold choice. Their scores are regression evidence only.",
        "The new pool contains 16,000 randomly sampled unused real reviews plus up to 2,000 unused short reviews. Its test distribution is deliberately short-review enriched; labels derive from stars (1–2 negative, 3 neutral, 4–5 positive). This is not a deployment prevalence estimate or independently human-labeled test benchmark.",
        f"Test class counts: {test.label.value_counts().to_dict()}; short (<=8 normalized whitespace words): {int((test.words <= 8).sum())}; long: {int((test.words > 8).sum())}.",
        f"Short-test class counts: {test[test.words <= 8].label.value_counts().to_dict()}. Short accuracy must be read alongside this class imbalance.",
        "The short-review audit is one assistant judgment on 36 training-only rows, not independent human adjudication. Ambiguous audited examples are excluded from fitting; no validation or test labels were edited. Remaining labels are rating proxies.",
        "Candidate training sizes differ from production. Comparisons against production combine data and configuration effects; baseline_fresh versus extra_short isolates added sampled short rows more closely.",
        "",
        "## Comparison",
        "",
        "| Model | Test accuracy | Macro-F1 | Short accuracy | Long accuracy | Regression /24 | Model MB | ms/review |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    flat = []
    for row in rows:
        m = row["test"]
        r = row["regression"]
        lines.append(
            f"| {row['name']} | {m['accuracy']:.2%} | {m['macro_f1']:.4f} | {m['short']['accuracy']:.2%} | {m['long']['accuracy']:.2%} | {round(r['accuracy'] * r['n'])}/24 | {row['size_bytes'] / 1e6:.2f} | {row['amortized_ms_per_review']:.3f} |"
        )
        flat.append(
            {
                "model": row["name"],
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "short_accuracy": m["short"]["accuracy"],
                "long_accuracy": m["long"]["accuracy"],
                "regression_correct": round(r["accuracy"] * r["n"]),
                "size_bytes": row["size_bytes"],
                "amortized_ms_per_review": row["amortized_ms_per_review"],
            }
        )
    lines += [
        "",
        "Timings are measured warm, CPU batch throughput divided by review count, not interactive single-request latency; exclude model loading. Transformer results use dynamic-int8 Linear layers, two CPU threads, batch 8, label-blind length sorting, and truncation to 256 tokens. Hybrid quality is scored by routing the stored model probabilities at the fixed threshold; conditional timing is measured separately by running TF-IDF on every row and the int8 transformer only on deferred rows. Dynamic quantization can produce small numerical differences with different batches. Model sizes exclude Python/dependency environments; the original downloaded full-precision weights are also retained separately.",
        "",
        "## Validation-only selection and calibration",
        "",
        f"Selected candidate before test/regression evaluation: **{selection['winner']}** (maximum validation macro-F1).",
        "",
        "| Candidate | Validation macro-F1 | Fit rows | Calibration rows | Training seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selection["candidates"]:
        c = row["config"]
        lines.append(
            f"| {row['name']} | {row['validation']['macro_f1']:.4f} | {c['fit_rows']} | {c['calibration_rows']} | {row['train_seconds']:.2f} |"
        )
    lines += [
        "",
        f"Validation abstention threshold: {selection['abstention_threshold']}. Rule: lowest prespecified threshold achieving 90% accepted accuracy. This threshold is diagnostic only and was not applied to production or the fixed 0.80 hybrid.",
        "```json",
        json.dumps(selection["abstention_validation"], indent=2),
        "```",
        "",
    ]
    calibration_frame = pd.read_csv(RUN / "calibration.csv")
    lines += [
        f"Short-only calibration class counts: {calibration_frame[calibration_frame.words <= 8].label.value_counts().to_dict()}. The small negative/neutral support limits this experiment; it is not evidence that all short-review calibration strategies fail.",
        "",
    ]
    calibration_path = RUN / "calibration_diagnostics.json"
    if calibration_path.exists():
        lines += [
            "",
            "## Calibration diagnostics",
            "",
            "Lower log loss, multiclass Brier sum, and 10-bin expected calibration error (ECE) are better. ECE depends on binning and class mix.",
            "",
            "| Model | Test log loss | Test Brier | Test ECE |",
            "|---|---:|---:|---:|",
        ]
        for item in json.loads(calibration_path.read_text()):
            if item["split"] == "test":
                lines.append(
                    f"| {item['name']} | {item['log_loss']:.4f} | {item['multiclass_brier_sum']:.4f} | {item['ece_10_bins']:.4f} |"
                )
    for row in rows:
        lines += [
            f"## {row['name']}",
            "",
            "| Class | Test precision | Test recall | Support |",
            "|---|---:|---:|---:|",
        ]
        for label in LABELS:
            c = row["test"]["per_class"][label]
            lines.append(
                f"| {label} | {c['precision']:.4f} | {c['recall']:.4f} | {int(c['support'])} |"
            )
        lines += [
            "",
            "Confusion matrices: rows actual, columns predicted; order negative, neutral, positive.",
            "",
            "Test:",
            "```json",
            json.dumps(row["test"]["confusion_matrix"]),
            "```",
            "Regression:",
            "```json",
            json.dumps(row["regression"]["confusion_matrix"]),
            "```",
        ]
        preds = pd.read_csv(RUN / f"{row['name']}_regression_predictions.csv")
        failures = preds[preds.label != preds.predicted]
        lines += ["", "Regression failures:"]
        lines += [
            f"- {r.review_text}: {r.label} → {r.predicted}" for r in failures.itertuples()
        ] or ["- None."]
        if row["name"].startswith("hybrid_"):
            base = next(r for r in rows if r["name"] == row["tfidf"])
            transformer = next(r for r in rows if r["name"] == "transformer")
            lines += ["", f"Deferred {row['deferred_fraction']:.2%} of test reviews."]
            for metric in ["accuracy", "macro_f1"]:
                lines.append(
                    f"- {metric}: difference versus TF-IDF {row['test'][metric] - base['test'][metric]:+.4f}; versus transformer {row['test'][metric] - transformer['test'][metric]:+.4f}."
                )
    if transformer_path.exists():
        info = json.loads((MODELS / "transformer/training_config.json").read_text())
        tr = next(r for r in rows if r["name"] == "transformer")
        lines += [
            "",
            "## Frozen transformer provenance",
            "",
            f"Model: {info['model']}; revision: {info['revision']}; license: {info['license']}.",
            f"[Pinned model card]({info['license_source']}). No fine-tuning was performed here; the upstream model was sentiment-fine-tuned on TweetEval. Its pretraining data overlap with this corpus cannot be independently ruled out.",
            f"Downloaded {info['download_bytes']:,} bytes; model load {tr['load_seconds']:.2f}s; {tr['truncated_test_rows']} test rows exceeded 256 tokens and were truncated.",
            "The reported transformer is the separate dynamic-int8 CPU candidate. Full-precision inference was stopped for runtime reasons before the 200-review progress marker; no full-precision aggregate scores are available. Quantization can change predictions. The runtime change was made without inspecting transformer test/regression metrics; see transformer_runtime_adjustment.json.",
        ]
    elif (RUN / "transformer_blocked.json").exists():
        lines += [
            "",
            "## Transformer blocked/skipped",
            "",
            "```json",
            (RUN / "transformer_blocked.json").read_text(),
            "```",
            "Hybrid evaluation is unavailable because transformer inference did not complete.",
        ]
    lines += ["", "## Artifact integrity", "", "| Candidate | SHA256 |", "|---|---|"]
    for row in rows:
        if "sha256" in row:
            lines.append(f"| {row['name']} | {row['sha256']} |")
    lines += [
        "",
        "Transformer file hashes and pinned revision are in its training_config.json; hybrids reference both component hashes and their fixed routing threshold. Full per-row probabilities, split hashes, configurations and metrics accompany this report.",
        "",
        "## Decision",
        "",
        "No model has been promoted. Keep production unchanged for now. If choosing a candidate for the next separately approved pilot, prefer the production-based 0.80 hybrid when macro-F1 and neutral recall matter more than overall accuracy and footprint. It has the highest measured test macro-F1 (0.6489) and 20/24 exposed regression cases correct, but its test accuracy falls from 85.58% to 84.21%, CPU batch throughput costs about 62.15 ms/review versus 0.29 ms, and deployment model assets grow from 1.70 MB to 245.22 MB. These are observed point estimates, not a claim of statistical significance or improved real-world performance.",
        "The hybrid beats production TF-IDF and the int8 transformer on macro-F1; it beats only the transformer on overall accuracy. Neutral recall rises from 10.80% in production to 49.77% in this hybrid, while other class trade-offs remain visible in the tables. The standalone transformer is not recommended on this rating-labeled benchmark. The added-short-review TF-IDF candidate has the highest overall test accuracy (85.99%) but only 12/24 regression cases correct; it also still misclassifies the original hate example.",
        "The int8 transformer and production hybrid classify 'Absolutely hate it' as negative with 79.10% probability. This is exposed regression evidence, not a selection criterion. Sparse linear models offer exact feature-score explanations; deferred transformer decisions do not. The quantization API emitted deprecation warnings, adding a maintenance consideration. A new independently human-labeled review benchmark, especially for neutral and short reviews, should gate any later deployment decision. Do not tune again on this now-inspected test set.",
        "Experiment 1 was a bounded pilot using previously unused real reviews from the existing sources and one assistant's 36-row training audit, not a completed independent human annotation effort. Experiment 3's short-only calibration had only 14 negative and 10 neutral examples. These limits prevent claiming that short-review augmentation or recalibration generally fails.",
    ]
    (RUN / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(flat).to_csv(RUN / "comparison.csv", index=False)
    write(RUN / "all_results.json", rows)
    guard()
    print(pd.DataFrame(flat).to_string(index=False))


if __name__ == "__main__":
    main()
