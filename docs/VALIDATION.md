# Validation record

Environment: Windows, Python 3.14.3, dependencies pinned in `requirements.txt`; full resolved runtime recorded in `requirements-lock.txt`. Real model trained 29 September 2026.

## Real-data experiment

- Input: both original local CSVs discovered under the user's Downloads directory; 568,454 + 28,332 rows. Source SHA-256 values are recorded in `models/metadata.json`.
- Cleaned unique data: 410,542 rows after invalid-text, conflicting-label and duplicate exclusions.
- Stratified sample: 30,000; training: 24,000; holdout: 6,000; seed 42.
- Five-fold training CV, TF-IDF refit inside every fold, macro F1 selection. Compared dummy, NB, logistic regression and LinearSVC. Optional RF path was not included in this experiment.
- Selected logistic regression (`C=2.0`, balanced class weights). Holdout accuracy 0.8371667, macro F1 0.6535452, weighted F1 0.8445005.
- Artifact: complete compressed pipeline, 1,711,347 bytes; trained only on the training split.
- Raw CSV files were read in place, not modified or added to Git. The original notebook/report are unchanged.

## Automated and runtime checks

Run `python -m pytest -q` using the project virtual environment. The final suite contains **31 passing tests**.

| Requested check | Evidence |
|---|---|
| App startup | Streamlit AppTest and actual local server on 127.0.0.1:8501 |
| Single prediction | AppTest plus real model smoke examples covering positive, neutral and negative |
| Blank input | Backend rejection and visible Streamlit error |
| Unusual characters | Unicode normalization, emoji-only rejection and unknown-vocabulary abstention |
| Very long input | Backend rejects >20,000 characters; Streamlit widget caps input length |
| CSV upload | Parser contract tests plus Streamlit FileUploader AppTest using the example CSV |
| Invalid CSV | Wrong schema, malformed quotes, invalid encoding, blank input, byte/row limits; invalid upload error and no stale prediction download |
| Prediction download | Exported bytes parsed back into CSV; Streamlit download widget exercised after upload |
| Model loading | Real artifact load, temporary artifact round trip, missing artifact handling and checksum corruption rejection |
| Preprocessing consistency | Raw/normalized predictions agree; saved pipeline reload agrees |
| Explanation | Local contributions plus intercept reconstruct the linear class score |
| Data leakage checks | Conflicting duplicate groups excluded; normalized text cannot overlap the train/holdout split |
| Dependency installation | New project virtual environment installed successfully; requirements dry-run satisfied; `pip check` reports no broken requirements |
| Streamlit compatibility | All seven pages execute through AppTest; local overview visibly rendered in the in-app browser |

## Browser and hosting limits

The local overview was visually checked and captured. The browser automation file-chooser event timed out for both exposed upload controls, so a native browser file-selection/download round trip was not verified. Upload and download application integration **was** tested through Streamlit's FileUploader/DownloadButton AppTest API, alongside CSV byte-level round-trip tests. This distinguishes an automation limitation from an observed application failure.

GitHub Actions is configured but has not run remotely. Streamlit Community Cloud deployment has not been performed or verified. Local Windows checks do not establish Linux hosting compatibility or operational reliability. Follow `DEPLOYMENT.md` for the remaining release check.

## Remaining evaluation limitations

Neutral F1 is 0.36. Labels are rating proxies. Exact deduplication does not eliminate near duplicates or shared products/authors/time between partitions. CV is used for tuning and selection, so its score is not an unbiased external benchmark. Probabilities are uncalibrated. No domain-shift, multilingual, robustness, fairness or latency SLA claims are made. Tests using synthetic examples validate functionality only; their scores are not portfolio results.
