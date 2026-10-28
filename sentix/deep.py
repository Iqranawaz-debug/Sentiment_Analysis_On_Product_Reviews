"""Optional, lazy, checksum-pinned int8 inference. No training or production writes."""

import hashlib
import json
import os
import re
import tempfile
import threading
import time
import urllib.request
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from sentix.predictor import LABELS, ROOT


def load_config():
    config = json.loads((ROOT / "configs/deep_mode.json").read_text(encoding="utf-8"))
    for name, variable in [
        ("url", "SENTIX_DEEP_MODEL_URL"),
        ("revision", "SENTIX_DEEP_REVISION"),
        ("sha256", "SENTIX_DEEP_SHA256"),
        ("cache_dir", "SENTIX_DEEP_CACHE"),
    ]:
        if variable in os.environ:
            config["artifact"][name] = os.environ[variable]
    if config["threshold"] != 0.80:
        raise DeepModeError("Deep Mode requires the approved 0.80 routing threshold.")
    if not re.fullmatch("[a-fA-F0-9]{64}", config["artifact"]["sha256"]):
        raise DeepModeError("Deep Mode needs a valid expected SHA-256.")
    # Dynamic int8 quantization takes its activation scale from the whole padded
    # batch, so batched scores depend on the neighbouring reviews (measured: up to
    # 0.25 probability and label flips). One review per forward pass keeps every
    # result independent and identical between Review Analyzer and Batch Analysis.
    if config["runtime"]["batch_size"] != 1:
        raise DeepModeError("Deep Mode must score one review at a time (batch_size 1).")
    return config


class DeepModeError(RuntimeError):
    """Actionable failure; never silently fall back while claiming Deep Mode."""


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise DeepModeError("Deep Mode refused a non-HTTPS redirect.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_https(url):
    return urllib.request.build_opener(HTTPSRedirectHandler()).open(url, timeout=30)


def checked_download(url, destination, expected, progress=None):
    destination = Path(destination)
    if destination.exists():
        if digest(destination) != expected:
            destination.unlink()
            raise DeepModeError(
                "Deep Mode cache failed SHA-256 verification and was deleted. Fast Mode was used."
            )
        return
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise DeepModeError("Deep Mode requires an HTTPS download URL.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        start = time.monotonic()
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".part", delete=False
        ) as out:
            temporary = Path(out.name)
            with open_https(url) as response:
                total = 0
                length = int(response.headers.get("Content-Length", 0))
                if progress:
                    progress(f"Downloading {destination.name}", 0.0)
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > 300 * 1024 * 1024 or time.monotonic() - start > 600:
                        raise DeepModeError("Deep Mode download exceeded its size/time limit.")
                    out.write(block)
                    if progress:
                        progress(
                            f"Downloading {destination.name}: {total / 1e6:.1f} MB",
                            min(total / length, 1.0) if length else 0.0,
                        )
        if digest(temporary) != expected:
            raise DeepModeError("Deep Mode download failed its SHA-256 check. No model was loaded.")
        temporary.replace(destination)
    except DeepModeError:
        raise
    except Exception as error:
        raise DeepModeError(
            "Deep Mode download failed. Check the configured URL and network, then retry."
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Int8Runtime:
    def __init__(self, cache_dir, config=None):
        self.cache_dir = Path(cache_dir)
        self.config = config or load_config()
        self.model = None
        self.lock = threading.RLock()

    def _load(self, progress=None):
        if self.model is not None:
            return
        weights = self.cache_dir / "state_dict.pt"
        artifact = self.config["artifact"]
        if not weights.exists() and (
            not re.fullmatch("[a-fA-F0-9]{40}", artifact["revision"])
            or artifact["revision"] not in artifact["url"]
        ):
            raise DeepModeError(
                "Deep Mode hosting is pending: configure its HTTPS URL and pinned commit SHA, "
                f"or place the verified weights in {self.cache_dir.name} (SENTIX_DEEP_CACHE "
                "selects the folder). Fast Mode was used."
            )
        checked_download(artifact["url"], weights, artifact["sha256"], progress)
        upstream = self.config["upstream"]
        asset_url = f"https://huggingface.co/{upstream['model']}/resolve/{upstream['revision']}"
        for name, sha in upstream["assets"].items():
            checked_download(f"{asset_url}/{name}", self.cache_dir / name, sha, progress)
        if progress:
            progress("Loading the verified int8 model", 1.0)
        try:
            import torch
            import transformers
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise DeepModeError(
                "Deep Mode needs optional dependencies. Install requirements-deep.txt alongside requirements.txt."
            ) from error
        settings = self.config["runtime"]
        if (
            torch.__version__.split("+")[0] != settings["torch"]
            or transformers.__version__ != settings["transformers"]
        ):
            raise DeepModeError(
                "Deep Mode requires the validated torch 2.14.1 and transformers 5.20.1 versions."
            )
        torch.set_num_threads(settings["threads"])
        config = AutoConfig.from_pretrained(self.cache_dir, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(self.cache_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_config(config).eval()
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(
            torch.load(weights, map_location="cpu", weights_only=True), strict=True
        )
        mapping = {int(k): v.lower() for k, v in config.id2label.items()}
        self.order = [next(k for k, v in mapping.items() if v == label) for label in LABELS]
        self.torch, self.tokenizer, self.model = torch, tokenizer, model.eval()

    def predict_proba(self, texts, progress=None):
        import numpy as np

        if not texts:
            return np.empty((0, 3))
        with self.lock:
            self._load(progress)
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
            result = np.empty((len(texts), 3))
            with self.torch.inference_mode():
                size = self.config["runtime"]["batch_size"]
                for start in range(0, len(texts), size):
                    positions = indices[start : start + size]
                    batch = [texts[i] for i in positions]
                    encoded = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config["runtime"]["max_length"],
                    )
                    result[positions] = self.torch.softmax(
                        self.model(**encoded).logits, dim=1
                    ).numpy()[:, self.order]
                    if progress:
                        progress(
                            f"Reading {min(start + size, len(texts))}/{len(texts)} deferred reviews",
                            min((start + size) / len(texts), 1.0),
                        )
            return result


@lru_cache(maxsize=2)
def cached_runtime(cache_dir, config_json):
    return Int8Runtime(cache_dir, json.loads(config_json))


class DeepPredictor:
    """Exactly the approved <0.80 gate; no separate neutral-class override."""

    mode = "Deep Mode"

    def __init__(self, production, runtime=None, progress=None):
        config = load_config()
        if production.metadata["artifact_sha256"] != config["production_sha256"]:
            raise DeepModeError("Deep Mode was validated only with the approved production model.")
        self.production = production
        self.metadata = production.metadata
        self.progress = progress
        self.threshold = config["threshold"]
        self.runtime = runtime or cached_runtime(
            str(ROOT / config["artifact"]["cache_dir"]),
            json.dumps(config, sort_keys=True),
        )

    def predict(self, text):
        return self.predict_batch([text])[0]

    def predict_batch(self, texts):
        texts = list(texts)
        results = self.production.predict_batch(texts)
        deferred = [
            i
            for i, r in enumerate(results)
            # The approved gate uses the largest calibrated probability, which the
            # Fast decision rule never changes.
            if r["status"] == "ok" and r.get("top_probability", r["confidence"]) < self.threshold
        ]
        for r in results:
            r["route"] = "fast" if r["status"] == "ok" else "unavailable"
        if deferred:
            try:
                probs = self.runtime.predict_proba(
                    [texts[i] for i in deferred], progress=self.progress
                )
            except Exception as error:
                detail = (
                    str(error)
                    if isinstance(error, DeepModeError)
                    else "Deep Mode could not load or run its model."
                )
                for i in deferred:
                    results[i]["route"] = "fast_fallback"
                    results[i]["warning"] = f"{detail} Showing the Fast Mode result."
                return results
            for i, p in zip(deferred, probs):
                label = LABELS[int(p.argmax())]
                results[i] = {
                    "status": "ok",
                    "label": label,
                    "confidence": float(p.max()),
                    "probabilities": dict(zip(LABELS, map(float, p))),
                    "route": "transformer",
                    "top_label": label,
                    "top_probability": float(p.max()),
                    "uncertainty": [],
                    "decision_rule": "transformer most probable class",
                }
        return results
