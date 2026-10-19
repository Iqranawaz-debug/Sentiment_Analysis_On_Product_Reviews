# Repository audit and implementation plan

Audit date: 29 September 2026. This records the original checkout **before implementation**. Findings use the four tracked files, all 23 notebook cells and their stored outputs, the 16-page report text, and its embedded comparison/workflow figures. Notebook cell numbers below are zero-based. Original artifacts are preserved. New-run evidence belongs in `models/metadata.json`, not in the historical notebook.

## Current architecture and assessment

Original flow: two local Colab CSVs → manual cleaning → 10,000-row sample → class downsampling → 80/20 stratified split → TF-IDF/model experiments → separate pickle exports → standalone Streamlit script. There are no supporting modules, tests, dependency declarations, model files, or deployment configuration in the original checkout. The original app does **not** retrain at startup; it immediately loads nonexistent files from an absolute Windows path. It has inference-only logic, rather than duplicated notebook training logic, but omits notebook cleaning.

This is a university project with useful ML experimentation and an early UI, not a reproducible portfolio release or production service. The preserved PDF explicitly identifies a university ML course and three team members. Preserve their attribution when presenting the project.

## A. What the project does well

- Uses practical sparse TF-IDF word unigrams/bigrams, a reasonable first approach for product reviews.
- Compares Naive Bayes, logistic regression, SVC, and Random Forest rather than claiming one algorithm is universally best.
- Uses sklearn Pipeline for TF-IDF plus estimator in the notebook; vocabulary fitting is inside the split/CV rather than fitted on the full dataset.
- Uses an 80/20 stratified split and seed 42 for sampling/splitting.
- Runs five-fold GridSearchCV; cross-validation and hyperparameter tuning **already exist**.
- Includes class distribution, review lengths, word cloud, confusion matrices, and classification reports.
- Maps 1–2 stars to negative, 3 to neutral, and 4–5 to positive. Neutral is a real third class, not a confidence threshold.
- Attempts model persistence and a Streamlit prediction UI. These are useful foundations despite implementation errors.

## B–J. Issues, impact, severity, and improvements

### B. Technical weaknesses

| Evidence / issue | Why it matters | Severity | Improvement |
|---|---|---|---|
| `app.py` loads `C:\Users\User\Documents\...` model and vectorizer files; neither exists in Git | Fresh checkout fails immediately | High | Repository-relative complete pipeline with an actionable load error |
| Cell 21 saves `model`, the last model loop variable, as `voting_model.pkl`; trained ensemble is `ensemble_pipe` | Artifact name and deployed estimator disagree | High | Persist selected complete pipeline and record estimator identity |
| Separately serialized vectorizer and estimator, no metadata | Artifacts can be mismatched; no version/data traceability | High | One pipeline, checksum, dependency versions, source hashes and configuration |
| Reused mutable `tfidf` object in several pipelines and repeated fitting | Makes notebook state fragile and hard to inspect | Medium | Fresh pipeline instances per model/search |

### C. Machine learning weaknesses

| Evidence / issue | Why it matters | Severity | Improvement |
|---|---|---|---|
| Cell 12: sample has 7,854 positive, 1,396 negative, 750 neutral; cell 14 reduces it to 750 per class | Discards 7,750 of 10,000 reviews and creates an artificially balanced holdout | High | Keep a stratified natural-distribution sample; compare class weights within CV |
| No duplicate or conflicting-text check | Same content can cross train/test or CV boundaries | High | Normalize and deduplicate before splitting; exclude conflicting groups, report counts |
| Product, author, source and time are removed | Random review split does not establish generalization to new products/authors/time | High | Document limitation; future grouped and temporal evaluations |
| GridSearchCV uses estimator-default accuracy | Majority performance can hide poor neutral/negative detection on natural data | High | Select by macro F1; show per-class precision/recall/F1 and weighted F1 |
| Test set repeatedly examined across cells 15–20 | Encourages model selection on the holdout | Medium | Tune/select using training CV only; evaluate selected estimator once |
| Training scores near 100% vs roughly 60% test for SVM/RF | Large generalization gap; high training accuracy is not evidence of quality | High | Report CV/holdout; regularize; avoid superiority claims |
| No dummy baseline | Accuracy lacks a meaningful reference | Medium | Include most-frequent baseline in same CV protocol |
| Cell 21 hard-codes scores 0.70, 0.72, 0.73, 0.71 | These values are not computed from that evaluation | High | Never use this results dictionary; generate all results from evaluation |
| Default RF and SVC probability fitting lack explicit random seeds | Repeated executions need not agree | Medium | Seed stochastic estimators and record environment |
| Rating-derived neutral can mean mixed experience | Proxy labels differ from directly annotated sentiment | Medium | Explain label semantics and examine neutral errors |
| No probability calibration evaluation | Confidence percentages can be overinterpreted | Medium | Label probabilities uncalibrated; add calibration only with separate validation |

### D. NLP weaknesses

| Evidence / issue | Why it matters | Severity | Improvement |
|---|---|---|---|
| Cell 10 removes NLTK stopwords including negation | “not good” loses important sentiment information | High | Keep stopwords and expand common negation contractions |
| ASCII-only deletion and punctuation concatenation | Loses accented words, numbers, and word boundaries | Medium | Unicode normalization and boundary-preserving cleaning |
| App sends raw text directly to vectorizer | Train/inference transformation mismatch | High | Put callable normalization inside saved TF-IDF pipeline |
| No blank-after-cleaning / unknown-vocabulary checks | Meaningless inputs still get authoritative predictions | Medium | Reject invalid text; abstain when all features are unknown |
| Word cloud alone has little diagnostic value | Frequent words do not explain classifier behavior | Low | Coefficient-based class terms and local contributions |
| No sarcasm, multilingual, aspect, or domain-shift validation | Broad social-media claims are not supported by product-review training | Medium | Scope to English product reviews; document limitations |

### E. Software engineering weaknesses

| Evidence / issue | Why it matters | Severity | Improvement |
|---|---|---|---|
| Notebook-only training with repeated loops | Hard to reproduce, test, or reuse | High | Focused preprocessing, data, train, evaluation, predict and UI modules |
| No requirements file; notebook upgrades libraries live and downloads NLTK data | Environment drifts and needs runtime network | High | Pin tested dependencies; no NLP corpus download required |
| No automated tests or CI | Artifact and UI regressions go undetected | High | Contract/unit tests, Streamlit AppTest and CI |
| No config, logs, provenance, artifact validation | Failures are hard to diagnose | Medium | JSON training config, progress logging, metadata and checksums |
| Legacy notebook imports XGBoost without using it; wordcloud dependency is undeclared | Unnecessary setup burden | Low | Keep legacy notebook historical; lean dependencies for current app |
| Seaborn palette deprecation appears in cell 8 output | Legacy plotting code may break on upgrades | Low | Preserve historical notebook; current app does not rely on that API |

### F. UI/UX weaknesses

| Evidence / issue | Why it matters | Severity | Improvement |
|---|---|---|---|
| Chart labels `[Positive, Negative, Neutral]` do not follow `classes_` | Probabilities are assigned to incorrect sentiments | High | Derive labels directly from model class order |
| Confidence converted into 0–5 product stars | A confident negative prediction can show five stars | High | Remove rating conversion entirely |
| Blank input silently does nothing; no load failure handling | User cannot diagnose failures | Medium | Clear validation and recovery messages |
| Single prediction only; no evaluation/provenance view | Reviewer cannot investigate performance or batch usefulness | Medium | Compact navigation for prediction, batch, data, performance and explanation |
| Forced CSS uses framework internals and hard-coded colors | Fragile styling and theme readability | Low | Native Streamlit theme and standard components |
| Ensemble description does not match saved model | Misrepresents what users are running | High | Display model identity from training metadata |

### G. Deployment weaknesses

| Evidence / issue | Why it matters | Severity | Improvement |
|---|---|---|---|
| Missing artifacts and dependencies; platform-specific paths | Streamlit Cloud cannot run checkout | High | Include small trusted artifact pair, requirements, relative paths and deployment guide |
| joblib loads on every Streamlit rerun | Avoidable latency and I/O | Medium | Cache resource with file modification keys |
| Full sources are absent; local copies discovered later total about 566 MB | Raw data would unnecessarily burden deployment | Medium | Train offline; do not commit raw data; deploy model only |
| No smoke tests or verified live URL | Deployment cannot honestly be claimed | Medium | Test local server; document hosted deployment as a separate unverified step |

### H. GitHub/README weaknesses

| Evidence / issue | Why it matters | Severity | Improvement |
|---|---|---|---|
| README mentions only Amazon Fine Food; cell 2 also reads Datafiniti May19 | Dataset provenance is incomplete | High | Document both schemas/sources and rating mapping |
| README/PDF say SVM/RF best after tuning | Saved notebook shows tuned LR 60.22%, NB/SVM 59.56%, RF 57.33% accuracy | High | Correct narrative; distinguish historical outputs from new results |
| PDF page 9 charts differ from notebook stored results; one axis says percent for 0–1 values | Evidence is internally inconsistent | Medium | Preserve report as historical; use generated metadata for current claims |
| No installation, run, architecture, screenshots, limitation or deployment instructions | Recruiter cannot quickly assess or reproduce project | High | Evidence-led README and linked detailed docs |
| README suggests preprocessing ensures fairness; report claims fairness checks without corresponding notebook evidence | Unsupported ethical/performance assurance | High | Remove claims; identify actual evaluated scope |
| Dataset versions/licenses are not recorded | Identical data and redistribution rights are uncertain | Medium | Hash local sources, link discovery pages, require license review before redistribution |

### I. LinkedIn portfolio weaknesses

| Evidence / issue | Why it matters | Severity | Improvement |
|---|---|---|---|
| No reproducible metric, live evidence or clear engineering story | Screenshots alone do not demonstrate sound ML work | High | Show measured macro F1, neutral weakness, pipeline and tests |
| Broad business claims lack deployment/user validation | Inflates impact beyond evidence | Medium | Frame feedback triage as an intended use, not measured business savings |
| Original project has three credited contributors | Solo presentation can obscure team contribution | Medium | Credit the original team and describe the later engineering work precisely |
| No curated screenshot sequence | Hard to communicate both product and technical work | Low | Use the seven-shot plan in `PORTFOLIO.md` |

### J. Production-readiness gaps

| Evidence / issue | Why it matters | Severity | Improvement |
|---|---|---|---|
| No monitoring, drift checks, alerting, access control, operational SLA or feedback labels | Cannot promise ongoing service quality | High | Keep portfolio scope explicit; add only for an actual operating use case |
| Pickle/joblib can execute code when loaded | Untrusted model artifacts are dangerous | High | Load repository-owned artifacts only; never accept uploaded models; checksum detects mismatch, not malicious publishers |
| No input/compute/upload limits | Very long text or large CSV can consume resources | Medium | Bound characters, bytes and batch rows |
| No export-safety design | CSV text can be interpreted as spreadsheet formulas | Medium | Escape leading formula markers in exported strings |
| No data-retention/privacy process | Public reviews may still contain personal information | Medium | Avoid saving uploaded reviews; exclude raw training data and usernames from release |

## Historical results, not a fresh benchmark

Cell 17 stored tuned accuracies: LR 60.22%, NB 59.56%, SVC 59.56%, RF 57.33%. Cell 19 ensemble report rounds accuracy to 0.62 and macro F1 to 0.61. The historical holdout has 450 examples, 150/class, from 2,250 balanced examples. These results are not directly comparable to the upgraded naturally distributed 30,000-row sample. No improvement percentage should be calculated across these different protocols.

## Prioritized implementation plan (presented before edits)

| Priority | Task / files | Reason and expected benefit | Difficulty |
|---|---|---|---|
| P0 | Complete pipeline: `src/preprocessing.py`, `src/train.py`, `src/predict.py`, `app.py` | Fix wrong artifact and preprocessing mismatch; consistent inference | Medium |
| P0 | Validation/deduplication/evaluation: `src/data.py`, `src/evaluation.py` | Limit content leakage and reveal minority-class performance | Medium |
| P1 | Model comparison: `src/train.py`, `config/training.json`, `models/` | Select by CV evidence and preserve untouched holdout | Medium |
| P1 | Dependencies/config/tests/CI: `requirements*.txt`, `tests/`, `.github/` | Reproducibility, regression detection, deployment compatibility | Medium |
| P2 | UI and portfolio: `src/ui.py`, `app.py`, `README.md`, `docs/` | Demonstrate batch utility, honest metrics, explanations and limitations | Medium |
| P3 | Future grouped/time splits, calibration and domain error analysis | Better external validity before operating a service | Hard |
| P3 | Optional transformer benchmark, only if error analysis and compute justify it | Test contextual improvements against strong sparse baseline | Hard |

## Architecture and model choices

Retain the notebook and report at their original paths to preserve useful work and links. Use small `src/` modules, `config/`, `models/`, `tests/`, `data/`, and `docs/`; no empty assets/notebooks/utils directories merely to resemble a template. Offline CLI training writes one model and one machine-readable report. Streamlit loads these without raw data.

Default candidates: most-frequent dummy, MultinomialNB, logistic regression, LinearSVC. Word unigrams/bigrams are interpretable and CPU-friendly. RF remains an opt-in comparison because sparse text and a large tree grid make it relatively costly; the historical notebook already explores it. XGBoost, deep learning, SHAP/LIME and a new voting ensemble are not justified without evidence that simpler models fail. Local linear contributions are sufficient for the default linear candidates.

Macro F1 is the selection metric because all three classes matter. Weighted F1 and accuracy describe sample-weighted behavior. Per-class precision/recall/F1 and the confusion matrix expose neutral/negative failure modes. ROC-AUC is omitted: label-level classification quality is the primary task, and uncalibrated multiclass margins do not support an honest probability-based AUC workflow without additional design.

## Reference guidance checked

- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html): consistent preprocessing, pipeline use and training/test separation.
- [scikit-learn persistence](https://scikit-learn.org/stable/model_persistence.html): trust and environment compatibility when loading artifacts.
- [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy): GitHub entry point, dependencies and Python selection.
