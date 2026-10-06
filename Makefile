PYTHON ?= python
.PHONY: train evaluate app test lint
train:
	$(PYTHON) -m src.train --config configs/default.yaml
evaluate:
	$(PYTHON) -m src.evaluate --reports reports --update-readme README.md
app:
	$(PYTHON) -m streamlit run app.py
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check .
