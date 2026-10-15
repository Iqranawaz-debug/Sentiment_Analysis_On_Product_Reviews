import re
import xml.etree.ElementTree as ET

import pytest

from sentix.game import score_round
from sentix.mascot import mood_state, render_senti


@pytest.mark.parametrize("mood", ["idle", "positive", "neutral", "negative", "unknown"])
@pytest.mark.parametrize("confidence", [0, 0.89, 0.90, 1, None])
def test_standalone_mascot(mood, confidence):
    rendered = render_senti(mood, confidence)
    assert rendered.startswith("<!doctype html>")
    ET.fromstring(re.search(r"<svg.*?</svg>", rendered, re.S).group())
    assert "prefers-reduced-motion" in rendered
    assert "<script" not in rendered


def test_intensity_boundary():
    assert mood_state("positive", 0.899) == "positive"
    assert mood_state("positive", 0.9) == "ecstatic"
    assert mood_state("negative", 0.899) == "negative"
    assert mood_state("negative", 0.9) == "very-upset"
    assert mood_state("neutral", 1) == "neutral"
    assert "&lt;script&gt;" in render_senti("idle", 0, "<script>")


def test_game_scoring():
    assert score_round("positive", "negative", "positive") == (1, 0)
    assert score_round("positive", "positive", "positive") == (1, 1)
    assert score_round("positive", "positive", None) == (0, 0)
