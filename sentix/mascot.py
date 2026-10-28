"""Original Senti character: a small mint seed with expressive antenna leaves."""

import html
import math
import os
import random

MESSAGES = {
    "idle": ["Paste a review and I'll tell you how people feel!"],
    "unsure": ["Hmm, I'm not sure about this one"],
    "positive": [
        "A little joy in every word!",
        "That sounds like a happy customer!",
        "This review made my day!",
    ],
    "negative": [
        "Oof. That experience needs some care.",
        "I hear a disappointed customer.",
        "Let's listen to what went wrong.",
    ],
    "neutral": [
        "Somewhere in the middle.",
        "A calm, everyday kind of review.",
        "No big celebration or storm here.",
    ],
}


def unsure_threshold() -> float:
    """UI-only threshold; invalid configuration falls back to 0.70."""
    try:
        value = float(os.environ.get("SENTIX_UNSURE_THRESHOLD", "0.70"))
        return value if math.isfinite(value) and 0 <= value <= 1 else 0.70
    except ValueError:
        return 0.70


def presentation_mood(result: dict) -> str:
    """Uncertainty changes Senti's face, never the saved model's decision."""
    if result.get("status") != "ok":
        return "idle"
    if result.get("uncertainty") or result["confidence"] < unsure_threshold():
        return "unsure"
    return result["label"]


def mood_state(mood: str, confidence: float) -> str:
    """Only actual high-confidence positive/negative predictions intensify."""
    confidence = (
        confidence if isinstance(confidence, (int, float)) and math.isfinite(confidence) else 0
    )
    if mood == "positive" and confidence >= 0.90:
        return "ecstatic"
    if mood == "negative" and confidence >= 0.90:
        return "very-upset"
    return mood if mood in MESSAGES else "idle"


def render_senti(
    mood: str, confidence: float, message: str | None = None, compact: bool = False
) -> str:
    """Return a responsive standalone SVG scene with reduced-motion support.

    `compact` draws a small companion beside results instead of a hero illustration.
    """
    size = (
        "svg{width:min(100%,150px);height:150px;overflow:visible}"
        ".bubble{font-size:13px;padding:8px 12px !important}.scene{padding:4px !important}"
        if compact
        else "svg{width:min(100%,270px);height:265px;overflow:visible}"
    )
    state = mood_state(mood, confidence)
    base = mood if mood in MESSAGES else "idle"
    bubble = html.escape(message or random.choice(MESSAGES[base]))
    eyes = '<g class="eyes"><ellipse cx="91" cy="123" rx="7" ry="10"/><ellipse cx="149" cy="123" rx="7" ry="10"/></g>'
    mouth = '<path d="M108 151 H132"/>'
    extra = ""
    if base == "unsure":
        eyes += '<path d="M78 104q12-12 24-2 M139 106h22" fill="none"/>'
        mouth = '<path d="M105 153q8-10 16 0t16 0"/>'
        extra = '<text x="183" y="77" font-size="35" fill="#8a5d00" stroke="none">?</text>'
    if base == "positive":
        eyes = '<path d="M80 125 Q91 107 102 125 M138 125 Q149 107 160 125" fill="none"/>'
        mouth = '<path d="M103 145 Q120 168 137 145"/>'
        extra = '<g class="sparkles" fill="#d9a72d" stroke="none"><path d="m35 72 4 10 10 4-10 4-4 10-4-10-10-4 10-4z"/><path d="m202 110 4 10 10 4-10 4-4 10-4-10-10-4 10-4z"/></g>'
    if state == "ecstatic":
        eyes = '<g fill="#8a5d00" stroke="none"><path d="m91 109 4 9 10 1-8 7 2 10-8-5-9 5 2-10-7-7 10-1z"/><path d="m149 109 4 9 10 1-8 7 2 10-8-5-9 5 2-10-7-7 10-1z"/></g>'
        extra += '<g class="confetti" stroke="#c58bdf"><path d="m50 35 4 8 M181 44l-4 8 M80 30l3 7 M160 25l-3 8"/></g>'
    if base == "negative":
        eyes += '<path d="m79 106 22-7 M139 99l22 7"/><g fill="white" stroke="none"><circle cx="94" cy="120" r="3"/><circle cx="152" cy="120" r="3"/></g>'
        mouth = '<path d="M106 155 Q120 140 134 155"/>'
        extra = '<path class="tear" d="M155 136q-12 18 0 18t0-18" fill="#5da9e0" stroke="none"/>'
    if state == "very-upset":
        eyes = '<circle cx="91" cy="125" r="7"/><circle cx="149" cy="125" r="7"/><path d="m79 106 22 7 M139 113l22-7"/>'
    return f'''<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><style>
    :root{{color-scheme:normal;background:transparent}}body{{margin:0;background:transparent;font:15px system-ui;text-align:center;color:#173e38}}
    .scene{{animation:arrive .35s ease-out;padding:12px 6px}}.bubble{{margin:auto;max-width:300px;padding:14px 18px;border-radius:20px;background:#edf8f2;color:#173e38;border:1px solid #99cbb8;line-height:1.5}}
    {size}.body{{transform-origin:120px 210px;animation:breathe 3s ease-in-out infinite}}
    .positive .body{{animation:bounce 1s ease-in-out infinite}}.ecstatic .body{{animation:bigbounce .65s ease-in-out infinite}}.negative .body{{animation:sway 2s ease-in-out infinite}}.very-upset .body{{animation:shake .35s ease-in-out infinite}}
    .eyes{{transform-box:fill-box;transform-origin:center;animation:blink 4s infinite}}.tear,.confetti{{animation:fall 1.6s infinite}}.sparkles{{animation:twinkle 1s infinite alternate}}
    @keyframes arrive{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
    @keyframes breathe{{50%{{transform:translateY(-3px)}}}}@keyframes bounce{{50%{{transform:translateY(-10px)}}}}@keyframes bigbounce{{50%{{transform:translateY(-20px) rotate(3deg)}}}}
    @keyframes sway{{25%{{transform:rotate(-3deg)}}75%{{transform:rotate(3deg)}}}}@keyframes shake{{25%{{transform:translateX(-2px)}}75%{{transform:translateX(2px)}}}}
    @keyframes blink{{0%,92%,100%{{transform:scaleY(1)}}96%{{transform:scaleY(.1)}}}}@keyframes fall{{to{{transform:translateY(24px);opacity:0}}}}@keyframes twinkle{{to{{opacity:.25}}}}
    @media(prefers-color-scheme:dark){{.bubble{{background:#193d36;color:#e1f6ec;border-color:#3d7565}}}}@media(prefers-reduced-motion:reduce){{*{{animation:none!important}}}}
    </style></head><body><div class="scene {state}" role="img" aria-label="Senti: {state}"><div class="bubble">{bubble}</div>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240"><ellipse cx="120" cy="221" rx="59" ry="8" fill="#70978933"/>
    <g class="body" stroke="#234d42" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">
    <path d="M119 66 Q83 51 96 28 Q126 31 119 66 M120 65Q124 28 151 35Q153 58 120 65" fill="#65b998"/>
    <path d="M66 178Q40 169 48 152 M174 178q26-9 18-26" fill="none"/>
    <path d="M91 207l-8 10 M149 207l8 10"/>
    <path d="M120 61C66 61 51 109 61 162Q64 210 120 211Q176 210 179 162C189 109 174 61 120 61Z" fill="#a5ddc4"/>
    <path d="M99 182q21 12 42 0" stroke="#75b99f" fill="none"/>
    <g fill="{"#ed7773" if state == "very-upset" else "#edaaa0"}" stroke="none" opacity=".8"><ellipse cx="77" cy="144" rx="12" ry="7"/><ellipse cx="163" cy="144" rx="12" ry="7"/></g>
    <g fill="#234d42">{eyes}</g><g fill="none">{mouth}</g>{extra}</g></svg></div></body></html>'''
