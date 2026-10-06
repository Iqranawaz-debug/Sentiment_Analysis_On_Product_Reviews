"""Leakage, split and negation contracts, independent of release artifacts."""

import pandas as pd
import pytest

from src.data import prepare_data, split_data
from src.preprocess import clean_text, without_stopwords


@pytest.mark.parametrize("normalize", [clean_text, without_stopwords])
def test_negation_survives(normalize):
    assert normalize("not good") == "not good"
    for word in ("not", "no", "nor", "never"):
        assert word in normalize(f"{word} good").split()
    assert "not" in normalize("isn't good").split()


def test_no_normalized_overlap_after_global_dedup():
    rows = [
        {"review_text": f"{label} review {i}", "label": label}
        for label in ("negative", "neutral", "positive")
        for i in range(100)
    ]
    rows += [{"review_text": "POSITIVE review 1!", "label": "positive"}]
    clean, audit = prepare_data(pd.DataFrame(rows))
    train, val, test = split_data(clean)
    assert audit["duplicate_rows_removed"] == 1
    assert [len(train), len(val), len(test)] == [210, 45, 45]
    assert not set(train.normalized) & set(test.normalized)
    assert not set(train.normalized) & set(val.normalized)
    assert not set(val.normalized) & set(test.normalized)
    assert all(set(part.label) == set(clean.label) for part in (train, val, test))
    again = split_data(clean)
    assert all(a.equals(b) for a, b in zip((train, val, test), again))


def test_readme_results_are_generated():
    import json
    from pathlib import Path

    from src.evaluate import results_markdown

    root = Path(__file__).resolve().parents[1]
    path = root / "reports/metrics.json"
    if not path.exists():
        pytest.skip("No measured release reports in this checkout")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    block = (
        (root / "README.md")
        .read_text(encoding="utf-8")
        .split("<!-- RESULTS:START -->")[1]
        .split("<!-- RESULTS:END -->")[0]
    )
    assert block.strip() == results_markdown(metrics).strip()
