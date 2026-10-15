"""Regression checks for edited text and honest low-confidence presentation."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from sentix.mascot import presentation_mood, render_senti, unsure_threshold


def test_edit_invalidates_then_reanalyzes():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=30
    ).run()
    next(b for b in app.button if b.label == "😍 Love").click().run()
    assert app.session_state.result["label"] == "positive"
    app.text_area[0].set_value("Absolutely hate it").run()
    assert app.session_state.result is None
    assert any("Review changed. Press Read the mood." in x.value for x in app.info)
    next(b for b in app.button if b.label == "Read the mood").click().run()
    assert not app.exception
    assert app.session_state.analyzed_text == "Absolutely hate it"
    assert presentation_mood(app.session_state.result) == "unsure"
    assert any(
        "Top two: Positive 64.9%" in x.value and "Negative 34.1%" in x.value for x in app.markdown
    )
    app.text_area[0].set_value("").run()
    assert app.session_state.result is None


def test_unsure_boundary_and_configuration(monkeypatch):
    monkeypatch.delenv("SENTIX_UNSURE_THRESHOLD", raising=False)
    result = {"status": "ok", "label": "positive", "confidence": 0.6999}
    assert presentation_mood(result) == "unsure"
    result["confidence"] = 0.70
    assert presentation_mood(result) == "positive"
    monkeypatch.setenv("SENTIX_UNSURE_THRESHOLD", ".80")
    assert presentation_mood(result) == "unsure"
    for value in ("bad", "nan", "2"):
        monkeypatch.setenv("SENTIX_UNSURE_THRESHOLD", value)
        assert unsure_threshold() == 0.70
    rendered = render_senti("unsure", 0.65)
    assert "Senti: unsure" in rendered and "?</text>" in rendered
