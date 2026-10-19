# Experiment evidence and reproduction

This run is intentionally immutable at the split/training/evaluation stages. Do
not remove its guards to tune after viewing test or regression results. A future
experiment needs a new run identifier and a predeclared protocol.

Commands executed from the repository root:

```powershell
python -m scripts.run_approved_experiments prepare
python -m scripts.run_approved_experiments train
python -m scripts.run_approved_experiments evaluate
python -m scripts.approved_transformer download
python -m scripts.approved_transformer evaluate
python -m scripts.check_approved_experiments
python -m scripts.report_approved_experiments
```

The training-only audit decisions were recorded between prepare and train, before
any candidate predictions. The transformer downloads and inference prerequisites
ran separately while the sparse experiments executed. Consult actual result files
to determine which stages completed; this command list is not itself evidence of
success.

The prepare stage verifies source checksums against the production evaluation,
reconstructs its 30,000-row sample, excludes those rows, screens near duplicates,
and saves group-disjoint splits. It writes checksums before any model fitting.
The train stage reads only fit, calibration, and validation data and saves every
trained model to a new candidate folder. It freezes its winner and abstention
threshold before evaluate can read the test set or the 24 regression cases.

Transformer weights are pinned to a Hugging Face commit, downloaded locally and
never fine-tuned. Full-precision CPU inference was too slow to complete; the
completed transformer benchmark uses a separately saved dynamic-int8 copy of
the Linear layers, with unchanged float32 embeddings. Batches are sorted by text
length without consulting labels, then restored to their original row order.
See `transformer_runtime_adjustment.json` for the runtime-only protocol amendment.
Its dependency installation is workspace-local:

```powershell
python -m pip install --target .experiment-deps torch transformers --only-binary=:all:
```

The transformer runner adds that directory after existing Python package paths;
it does not replace the production environment. Exact installed versions and
artifact hashes are saved with the run. The two hybrids reuse the same downloaded
weights and reference their component hashes instead of duplicating a 501 MB
weight file. The routing threshold is fixed at 0.80, as requested.

Files:

- `REPORT.md`: comparison, per-class precision/recall, confusion matrices, and all
  regression failures for every completed model/policy.
- `comparison.csv`, `all_results.json`: machine-readable results.
- `*_predictions.csv`: actual per-review labels and probabilities.
- `protocol.json`: split definition, source and split checksums, selection rules.
- `selection.json`: validation evidence and the frozen selection.
- `training_label_audit.csv`, `audit_decisions.json`: training-only label audit.
- `calibration_diagnostics.json`: log loss, Brier sum, and ECE.
- `verification.json`: executed integrity and consistency checks.
- `production_hashes.json`: production file checksums captured before work.
- `inference-requirements.txt`: exact workspace-local inference dependencies.
- `transformer_remote_integrity.json`: downloaded weights match the pinned
  repository's LFS checksum.

Test labels remain rating-derived. Assistant audit judgments on training data
are not independent human adjudication, and duplicate screening is not a proof
of product-, user-, or semantic-level independence. Further hyperparameter changes
must not use this now-inspected test set as a fresh final benchmark.
