# Upgrade verification

## Implementation decisions

- Preserve the existing cached offline loading, integrity checks, input validation and
  formula-safe exports. Reject old uncalibrated artifact schemas with retraining guidance.
- Resolve the CV/validation selection ambiguity by fixed-setting family selection on
  fitting-only training CV, then validation-only regularization tuning. The calibration
  subset is reserved inside training before any fitting or selection.
- Keep the serving artifact a single sklearn Pipeline. Its final estimator is a sigmoid
  CalibratedClassifierCV around the frozen linear classifier; explanations access the
  underlying coefficients and explicitly describe pre-calibration scores.
- Use an explicit capped real-data run for the delivered evidence. Defaults are uncapped;
  fresh-output reproduction and transformer configs are supplied. Training refuses to
  overwrite an existing metrics report to discourage accidental repeated test evaluation.
- Keep raw reviews, complete test predictions and confident error samples local. Aggregate
  reports and the model are tracked. The Error Explorer explains missing local samples.
- Preserve unrelated pre-existing audit/deployment/portfolio documents in the working
  tree; they describe the older implementation and are not current evidence.

## Completed checks

- pytest passed, including end-to-end synthetic training, report/prediction consistency,
  generated README consistency, negation, disjoint splits, calibrated explanations,
  artifact corruption, CSV safety and Streamlit AppTest flows.
- Ruff, dependency consistency and Git whitespace checks passed.
- Real Amazon-data training completed, with all final metrics generated in `reports/`.
  Test predictions were reused for source metrics, confusion matrices, reliability and
  errors. No second test pass or test-based retuning was performed.
- Local Streamlit app was opened in a browser; its prediction and comparison views were
  inspected and genuine screenshots saved under `docs/screenshots/`.
- The optional Trainer path follows the documented Hugging Face API and is import-gated.

## Not verified

- Full-corpus training: the delivered run explicitly uses the fast-run sample cap.
- DistilBERT fine-tuning: optional dependencies/weights were not downloaded or executed;
  no transformer benchmark is published.
- Docker execution: Docker is not installed on the implementation host.
- Hosted GitHub Actions and deployment: local equivalent checks passed; changes were
  committed locally, not pushed or deployed.

## Interpretation

Read the generated per-class and source metrics, not accuracy alone. Neutral recall
is weak, and the consumer-product source slice is small. Sigmoid calibration can
improve probability interpretation without improving argmax macro-F1. The screenshot
of a mixed review is a visible failure example, not a cherry-picked success.
