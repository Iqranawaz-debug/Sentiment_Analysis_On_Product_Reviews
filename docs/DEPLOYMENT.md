# Local execution and Streamlit Community Cloud

## Supported environment

The measured release uses Python 3.14.3. Use Python 3.14 for local execution and hosting. `requirements.txt` pins all directly used runtime libraries, including the numerical libraries required by the serialized model. `requirements-lock.txt` records the full resolved Windows runtime; `requirements-dev.txt` adds pytest. CI is configured for Linux/Python 3.14; local Windows verification does not prove a remote CI run passed.

## Run locally (PowerShell, repository root)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Activation is optional; these commands avoid PowerShell execution-policy changes. On Linux/macOS replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Retrain

Inference only needs the tracked `models/pipeline.joblib` and `models/metadata.json`. Training needs both source CSVs. Put your authorized copies in `data/raw/` (ignored by Git), then run:

```powershell
.\.venv\Scripts\python.exe -m src.train --data data/raw/Amazon_Food_Reviews.csv data/raw/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv
```

To experiment without replacing the release model:

```powershell
.\.venv\Scripts\python.exe -m src.train --data data/raw/Amazon_Food_Reviews.csv --config config/training.json --output models/experiment
```

Edit a copy of the JSON configuration to change sample size, folds, seed or the optional Random Forest comparison. `sample_size: 0` uses all cleaned rows and can require substantially more time/memory. The default is 30,000 stratified reviews. Do not tune repeatedly on the saved holdout results. A new release should use a predeclared protocol and, when practical, a fresh external holdout.

## Streamlit Community Cloud

1. Review and push the application code, requirements, `src/`, `.streamlit/config.toml`, and both trusted model files to your GitHub repository. Raw datasets are unnecessary for serving.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, select your repository and branch, and set the entry point to **app.py**.
3. In advanced settings select **Python 3.14**, matching the trained artifact environment, then deploy.
4. Check the build logs, overview, one review, batch upload/download, and performance page. Add the verified public URL to README and the LinkedIn draft only after successful deployment.

Community Cloud deployment is a dashboard action; there is no project-specific deployment CLI command to invent. For a local deployment smoke test:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless=true --server.port=8501
```

No hosted deployment, Git push or live URL is claimed by this local implementation. See [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency layout](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

## Artifact and operational precautions

Only load model files produced by this project or a trusted maintainer. Joblib files are executable serialization; a matching SHA-256 checks integrity, not trust. The app rejects dependency-version and checksum mismatches. Restore a matching pair or retrain after environment changes. A failed load shows a recovery message and logs the exception without logging uploaded review text.

Uploaded reviews are not deliberately written to disk by application code. Results live in session execution memory. This does not constitute an audited hosting-provider retention guarantee. Upload limits are 5 MB / 1,000 rows / 20,000 characters per review. CSV export prefixes potentially executable text with an apostrophe. Authentication, rate limiting across sessions, operational monitoring and drift alerts are future service requirements.
