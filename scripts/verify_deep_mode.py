"""Real Streamlit CSV workflow, local approved artifact, no cached predictions."""

import argparse
import faulthandler
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from streamlit.testing.v1 import AppTest

from sentix.deep import Int8Runtime, digest, load_config
from sentix.predictor import LABELS, load_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default="reports/deep_mode_e2e")
    args = parser.parse_args()
    # Development-only dependencies; deployed app installs requirements-deep.txt.
    sys.path.append(str(ROOT / ".experiment-deps"))
    report_dir = ROOT / args.report_dir
    report_dir.mkdir(exist_ok=False)
    faulthandler.dump_traceback_later(180, repeat=True)
    original_predict = Int8Runtime.predict_proba

    def logged_predict(self, texts, progress=None):
        def log(message, fraction):
            print(message, flush=True)
            if progress:
                progress(message, fraction)

        return original_predict(self, texts, progress=log)

    Int8Runtime.predict_proba = logged_predict
    before = {p.name: digest(p) for p in (ROOT / "models").iterdir() if p.is_file()}
    config = load_config()
    cache = ROOT / ".cache/deep-mode-e2e"
    cache.mkdir(parents=True, exist_ok=True)
    source = ROOT / "models/candidates/approved_20261016"
    shutil.copyfile(source / "transformer_int8/state_dict.pt", cache / "state_dict.pt")
    for name in config["upstream"]["assets"]:
        shutil.copyfile(source / "transformer" / name, cache / name)
    assert digest(cache / "state_dict.pt") == config["artifact"]["sha256"]
    os.environ["SENTIX_DEEP_CACHE"] = str(cache)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=1200).run()
    assert not app.exception and app.radio[0].value == "Fast Mode"
    app.radio[0].set_value("Deep Mode").run()
    assert not app.exception
    results = {}
    production = load_model()
    for name, filename in [
        ("test", "test.csv"),
        ("regression", "production_regression_predictions.csv"),
    ]:
        frame = pd.read_csv(ROOT / "reports/approved_20261016" / filename)
        csv = frame[["review_text"]].to_csv(index=False).encode()
        app.file_uploader[0].set_value((f"{name}.csv", csv, "text/csv")).run()
        start = time.perf_counter()
        print("Running actual Streamlit CSV analysis:", name, len(frame), flush=True)
        next(b for b in app.button if b.label == "Analyze CSV").click().run(timeout=1200)
        assert not app.exception, app.exception
        actual = app.session_state.bulk_result
        assert len(actual) == len(frame)
        actual.to_csv(report_dir / f"{name}_actual.csv", index=False)
        expected_status = [r["status"] for r in production.predict_batch(frame.review_text.tolist())]
        assert actual.status.tolist() == expected_status
        assert not (actual.route == "fast_fallback").any(), actual.warning.unique()
        results[name] = {
            "n": len(frame),
            "accuracy": accuracy_score(frame.label, actual.sentiment),
            "macro_f1": f1_score(frame.label, actual.sentiment, labels=list(LABELS), average="macro"),
            "correct": int((frame.label.to_numpy() == actual.sentiment.to_numpy()).sum()),
            "confusion_matrix": confusion_matrix(
                frame.label, actual.sentiment, labels=[*LABELS, "unavailable"]
            ).tolist(),
            "confusion_matrix_order": [*LABELS, "unavailable"],
            "unavailable_counted_incorrect": int((actual.status != "ok").sum()),
            "transformer_rows": int((actual.route == "transformer").sum()),
            "seconds": time.perf_counter() - start,
        }
        print(name, results[name], flush=True)
    # Tolerances declared before the run: batch-dependent int8 numerical variation.
    results["acceptance"] = {
        "reference_macro_f1": 0.6489,
        "absolute_macro_f1_tolerance": 0.005,
        "required_regression_correct": 20,
        "macro_f1_pass": abs(results["test"]["macro_f1"] - 0.6489) <= 0.005,
        "regression_pass": results["regression"]["correct"] == 20,
    }
    after = {p.name: digest(p) for p in (ROOT / "models").iterdir() if p.is_file()}
    assert after == before
    results["production_hashes_before"] = before
    results["production_hashes_after"] = after
    results["artifact_sha256"] = digest(cache / "state_dict.pt")
    results["download_test"] = (
        "Local approved artifact staged explicitly; real hosted HTTPS download pending URL. HTTPS download/failure behavior covered by unit tests."
    )
    (report_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    assert results["acceptance"]["macro_f1_pass"] and results["acceptance"]["regression_pass"], (
        results["acceptance"]
    )
    print("Deep Mode end-to-end acceptance passed", flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
