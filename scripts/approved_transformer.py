"""Download a pinned frozen transformer and benchmark it without fine-tuning."""

import argparse
import json
import sys
import time
import urllib.request

from scripts.run_approved_experiments import (
    LABELS,
    MODELS,
    ROOT,
    RUN,
    guard,
    metrics,
    save_predictions,
    sha256_file,
    write,
)

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
DEST = MODELS / "transformer"
FILES = [
    "README.md",
    "config.json",
    "merges.txt",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "vocab.json",
]


def download():
    DEST.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    manifest = []
    try:
        for name in FILES:
            print("Downloading", name, flush=True)
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
            path = DEST / name
            with urllib.request.urlopen(url, timeout=45) as response, path.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
                    if time.perf_counter() - start > 600:
                        raise TimeoutError("Model download exceeded ten-minute budget")
            manifest.append(
                {"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
        write(
            DEST / "training_config.json",
            {
                "model": MODEL,
                "revision": REVISION,
                "license": "cc-by-4.0",
                "license_source": f"https://huggingface.co/{MODEL}/blob/{REVISION}/README.md",
                "fine_tuning": False,
                "download_seconds": time.perf_counter() - start,
                "files": manifest,
                "download_bytes": sum(f["bytes"] for f in manifest),
                "max_length": 256,
                "batch_size": 8,
                "cpu_threads": 2,
                "preprocessing": "model-card URL and username placeholders",
            },
        )
    except Exception as error:
        write(
            RUN / "transformer_blocked.json",
            {
                "error": repr(error),
                "completed_files": manifest,
                "model": MODEL,
                "revision": REVISION,
                "advertised_weights_bytes": 501045531,
                "license": "cc-by-4.0",
            },
        )
        raise


def evaluate():
    import numpy as np
    import pandas as pd

    # Prefer existing environment packages; additional inference dependencies stay local.
    sys.path.append(str(ROOT / ".experiment-deps"))
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    guard()
    assert not (RUN / "transformer_results.json").exists()
    torch.set_num_threads(2)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(DEST, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(DEST, local_files_only=True).eval()
    print("Loaded frozen model; creating separate dynamic-int8 candidate", flush=True)
    quantized_dir = MODELS / "transformer_int8"
    quantized_dir.mkdir(exist_ok=False)
    model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(model.state_dict(), quantized_dir / "state_dict.pt")
    quantized_hash = sha256_file(quantized_dir / "state_dict.pt")
    write(
        quantized_dir / "training_config.json",
        {
            "source_model": MODEL,
            "revision": REVISION,
            "fine_tuning": False,
            "quantization": "torch dynamic qint8 Linear layers; float32 embeddings",
            "sha256": quantized_hash,
            "tokenizer_and_config": str(DEST.relative_to(ROOT)),
            "source_manifest_sha256": sha256_file(DEST / "training_config.json"),
            "torch": torch.__version__,
            "cpu_threads": 2,
            "batch_size": 8,
            "max_length": 256,
            "batching": "label-blind text-length sorting, restored original order",
        },
    )
    quantized_size = (quantized_dir / "state_dict.pt").stat().st_size + sum(
        (DEST / f).stat().st_size for f in FILES if f != "pytorch_model.bin"
    )
    load_seconds = time.perf_counter() - start
    mapping = {int(k): v.lower() for k, v in model.config.id2label.items()}
    order = [next(k for k, v in mapping.items() if v == label) for label in LABELS]
    truncated = 0

    def predict(texts, progress=False):
        nonlocal truncated
        result = np.empty((len(texts), 3))
        texts = [
            " ".join(
                "@user"
                if w.startswith("@") and len(w) > 1
                else "http"
                if w.startswith("http")
                else w
                for w in t.split(" ")
            )
            for t in texts
        ]
        indices = np.argsort([len(t) for t in texts], kind="stable")
        with torch.inference_mode():
            for start in range(0, len(texts), 8):
                positions = indices[start : start + 8]
                batch = [texts[i] for i in positions]
                truncated += sum(len(tokenizer.encode(t)) > 256 for t in batch)
                encoded = tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=256
                )
                result[positions] = torch.softmax(model(**encoded).logits, dim=1).numpy()[:, order]
                if progress and start % 200 == 0:
                    print("Transformer", start, "/", len(texts), flush=True)
        return result

    test = pd.read_csv(RUN / "test.csv")
    regression = pd.read_csv(RUN / "production_regression_predictions.csv").rename(
        columns={"predicted": "unused"}
    )
    predict(["A warmup example."])
    truncated = 0
    start = time.perf_counter()
    p = predict(test.review_text.tolist(), True)
    elapsed = time.perf_counter() - start
    truncated_test = truncated
    rp = predict(regression.review_text.tolist())
    save_predictions("transformer", test, p, regression, rp)
    info = json.loads((DEST / "training_config.json").read_text())
    row = {
        "name": "transformer",
        "sha256": quantized_hash,
        "sha256_kind": "dynamic-int8 state_dict.pt",
        "quantization": "dynamic qint8 Linear layers; no fine-tuning",
        "model": MODEL,
        "revision": REVISION,
        "license": info["license"],
        "size_bytes": quantized_size,
        "download_bytes": info["download_bytes"],
        "load_seconds": load_seconds,
        "test_seconds": elapsed,
        "amortized_ms_per_review": 1000 * elapsed / len(test),
        "truncated_test_rows": truncated_test,
        "test": metrics(test, p),
        "regression": metrics(regression, rp),
    }
    selection = json.loads((RUN / "selection.json").read_text())
    rows = [row]
    # Evaluate both the production fallback and the validation-selected TF-IDF fallback.
    for name in ["production", selection["winner"]]:
        tf = pd.read_csv(RUN / f"{name}_test_predictions.csv")[LABELS].to_numpy()
        rf = pd.read_csv(RUN / f"{name}_regression_predictions.csv")[LABELS].to_numpy()
        defer = tf.max(axis=1) < 0.80
        defer_r = rf.max(axis=1) < 0.80
        hp = tf.copy()
        hp[defer] = p[defer]
        hr = rf.copy()
        hr[defer_r] = rp[defer_r]
        hybrid_name = f"hybrid_{name}"
        save_predictions(hybrid_name, test, hp, regression, hr)
        # Time actual conditional inference separately rather than estimating hybrid cost.
        import joblib

        from scripts.run_approved_experiments import probabilities

        path = (
            ROOT / "models/pipeline.joblib"
            if name == "production"
            else MODELS / name / "pipeline.joblib"
        )
        sparse = joblib.load(path)
        start = time.perf_counter()
        tf_timed = probabilities(sparse, test.review_text)
        deferred = tf_timed.max(axis=1) < 0.80
        if deferred.any():
            predict(test.review_text[deferred].tolist(), True)
        seconds = time.perf_counter() - start
        hybrid = {
            "name": hybrid_name,
            "tfidf": name,
            "threshold": 0.80,
            "deferred_fraction": float(defer.mean()),
            "test": metrics(test, hp),
            "regression": metrics(regression, hr),
            "test_seconds": seconds,
            "amortized_ms_per_review": 1000 * seconds / len(test),
            "size_bytes": quantized_size + path.stat().st_size,
        }
        rows.append(hybrid)
        out = MODELS / hybrid_name
        out.mkdir(exist_ok=False)
        write(
            out / "training_config.json",
            {
                "tfidf_path": str(path.relative_to(ROOT)),
                "tfidf_sha256": sha256_file(path),
                "transformer_revision": REVISION,
                "transformer_manifest_sha256": sha256_file(DEST / "training_config.json"),
                "quantized_transformer_sha256": quantized_hash,
                "threshold": 0.80,
                "training": "none; deterministic routing",
            },
        )
        hybrid["sha256"] = sha256_file(out / "training_config.json")
        hybrid["sha256_kind"] = "routing config referencing component hashes"
    write(RUN / "transformer_results.json", rows)
    guard()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["download", "evaluate"])
    globals()[parser.parse_args().stage]()
