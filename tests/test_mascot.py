import re
import xml.etree.ElementTree as ET

import pytest

from sentix.game import score_round
from sentix.mascot import mood_state, render_senti, senti_svg


@pytest.mark.parametrize(
    "mood", ["idle", "welcome", "thinking", "unsure", "positive", "neutral", "negative", "unknown"]
)
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


def test_expressions_follow_real_states():
    """Each face matches an app state; decorations never overstate a result."""
    assert 'class="dots"' in render_senti("thinking", 0)
    assert 'class="wave"' in render_senti("welcome", 0)
    negative = render_senti("negative", 0.95)
    assert "tear" not in negative and "shake" not in negative.split("</style>")[1]
    assert 'class="heart"' in render_senti("positive", 0.95)
    assert 'class="heart"' not in render_senti("positive", 0.85)
    assert 'class="sparkles"' not in render_senti("unsure", 0.6)
    for mood in ("welcome", "thinking", "positive", "neutral", "negative", "unsure"):
        ET.fromstring(senti_svg(mood, 0.95))
        assert "<script" not in senti_svg(mood)
    assert 'aria-hidden="true"' in render_senti("welcome", 0)
    assert 'role="img"' in senti_svg("welcome")
