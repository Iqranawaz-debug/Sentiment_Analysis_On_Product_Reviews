"""Routing, safe download failure, and default/fallback UI behavior."""

import hashlib
import io
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from sentix import deep


class Production:
    metadata = {"artifact_sha256": deep.load_config()["production_sha256"]}

    def predict_batch(self, texts):
        return [
            dict(
                status="ok",
                label="neutral",
                confidence=float(t),
                probabilities={
                    "negative": (1 - float(t)) / 2,
                    "neutral": float(t),
                    "positive": (1 - float(t)) / 2,
                },
            )
            for t in texts
        ]


def test_exact_gate_including_confident_neutral():
    class Runtime:
        calls = []

        def predict_proba(self, texts, progress=None):
            self.calls.append(texts)
            return np.array([[0.9, 0.05, 0.05]] * len(texts))

    runtime = Runtime()
    result = deep.DeepPredictor(Production(), runtime).predict_batch(["0.7999", "0.8", "0.95"])
    assert runtime.calls == [["0.7999"]]
    assert [r["route"] for r in result] == ["transformer", "fast", "fast"]
    assert result[1]["label"] == "neutral"


class Response(io.BytesIO):
    headers = {"Content-Length": "3"}


def test_checksum_failure_deletes_download_and_bad_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(deep, "open_https", lambda url: Response(b"bad"))
    destination = tmp_path / "state_dict.pt"
    with pytest.raises(deep.DeepModeError, match="SHA-256"):
        deep.checked_download("https://example.test/model", destination, "0" * 64)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))
    destination.write_bytes(b"bad")
    with pytest.raises(deep.DeepModeError, match="deleted"):
        deep.checked_download("https://example.test/model", destination, "0" * 64)
    assert not destination.exists()


def test_verified_download_progress_and_cache_reuse(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(deep, "open_https", lambda url: calls.append(url) or Response(b"yes"))
    updates = []
    path = tmp_path / "state_dict.pt"
    sha = hashlib.sha256(b"yes").hexdigest()
    deep.checked_download("https://example.test/model", path, sha, lambda *x: updates.append(x))
    deep.checked_download("https://example.test/model", path, sha)
    assert len(calls) == 1 and updates[-1][1] == 1.0
    assert path.read_bytes() == b"yes"


def test_bad_checksum_falls_back_without_loading_model(tmp_path):
    weights = tmp_path / "state_dict.pt"
    weights.write_bytes(b"untrusted bytes")
    runtime = deep.Int8Runtime(tmp_path)
    actual = deep.DeepPredictor(Production(), runtime).predict("0.6")
    assert actual["route"] == "fast_fallback"
    assert actual["label"] == "neutral" and actual["confidence"] == 0.6
    assert "SHA-256" in actual["warning"]
    assert not weights.exists() and runtime.model is None


def test_https_only(tmp_path):
    with pytest.raises(deep.DeepModeError, match="HTTPS"):
        deep.checked_download("http://localhost/model", tmp_path / "model", "0" * 64)
    with pytest.raises(deep.DeepModeError, match="redirect"):
        deep.HTTPSRedirectHandler().redirect_request(
            None, None, 302, "", {}, "http://example.test/model"
        )


def test_offline_fallback_preserves_fast_result(tmp_path, monkeypatch):
    def offline(url):
        raise URLError("offline")

    monkeypatch.setattr(deep, "open_https", offline)
    config = deep.load_config()
    config["artifact"].update(
        revision="a" * 40, url="https://example.test/" + "a" * 40 + "/state_dict.pt"
    )
    runtime = deep.Int8Runtime(tmp_path, config)
    actual = deep.DeepPredictor(Production(), runtime).predict("0.6")
    assert actual["label"] == "neutral" and actual["confidence"] == 0.6
    assert actual["route"] == "fast_fallback"
    assert "download failed" in actual["warning"]
    assert not list(tmp_path.glob("*.part"))


def test_ui_defaults_fast_and_missing_deep_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTIX_DEEP_CACHE", str(tmp_path))
    monkeypatch.delenv("SENTIX_DEEP_MODEL_URL", raising=False)
    monkeypatch.delenv("SENTIX_DEEP_REVISION", raising=False)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60
    ).run()
    assert app.radio(key="analysis_mode").value == "Fast Mode"
    app.text_area[0].set_value("Absolutely hate it").run()
    next(b for b in app.button if b.label == "Read the mood").click().run()
    fast = app.session_state.result
    app.radio(key="analysis_mode").set_value("Deep Mode").run()
    assert app.session_state.result is None
    next(b for b in app.button if b.label == "Read the mood").click().run()
    assert not app.exception
    assert app.session_state.result["probabilities"] == fast["probabilities"]
    assert app.session_state.result["route"] == "fast_fallback"
    assert any("hosting is pending" in w.value for w in app.warning)
    app.radio(key="analysis_mode").set_value("Fast Mode").run()
    assert app.session_state.result is None
