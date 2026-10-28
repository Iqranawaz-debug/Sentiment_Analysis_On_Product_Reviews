"""Original Senti character: a small mint seed with expressive antenna leaves."""

import base64
import html
import math
import os
import random

MESSAGES = {
    "idle": ["Paste a review and I'll tell you how people feel!"],
    "welcome": ["Hi! Paste a review and I'll read it with the model."],
    "thinking": ["Reading carefully…"],
    "unsure": ["Hmm, I'm not sure about this one"],
    "positive": [
        "A little joy in every word!",
        "That sounds like a happy customer!",
        "This review made my day!",
    ],
    "negative": [
        "This customer had a rough time.",
        "Let's look at what went wrong.",
        "Worth a closer look and a kind reply.",
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


def face(mood: str, state: str) -> tuple[str, str, str]:
    """Eyes, mouth and extras for each state. Expressions follow real app states only:
    welcome/idle before analysis, thinking during inference, then the result's mood."""
    eyes = (
        '<g class="eyes"><ellipse cx="91" cy="123" rx="7" ry="10"/>'
        '<ellipse cx="149" cy="123" rx="7" ry="10"/></g>'
    )
    mouth = '<path d="M108 151 H132"/>'
    extra = ""
    if mood == "welcome":
        mouth = '<path d="M106 147 Q120 160 134 147"/>'
    if mood == "thinking":
        eyes = (
            '<g class="eyes"><ellipse cx="95" cy="117" rx="6" ry="8"/>'
            '<ellipse cx="153" cy="117" rx="6" ry="8"/></g>'
        )
        mouth = '<path d="M110 153 q10 4 20 0"/>'
        extra = (
            '<g class="dots" fill="#0E6B61" stroke="none"><circle cx="186" cy="62" r="5"/>'
            '<circle cx="202" cy="50" r="6"/><circle cx="220" cy="36" r="7"/></g>'
        )
    if mood == "unsure":
        eyes += '<path d="M78 104q12-12 24-2 M139 106h22" fill="none"/>'
        mouth = '<path d="M105 153q8-10 16 0t16 0"/>'
        extra = '<text x="183" y="77" font-size="35" fill="#8a5d00" stroke="none">?</text>'
    if mood == "positive":
        eyes = '<path d="M80 125 Q91 107 102 125 M138 125 Q149 107 160 125" fill="none"/>'
        mouth = '<path d="M103 145 Q120 168 137 145"/>'
        extra = (
            '<g class="sparkles" fill="#E0A63A" stroke="none">'
            '<path d="m35 72 4 10 10 4-10 4-4 10-4-10-10-4 10-4z"/>'
            '<path d="m202 110 4 10 10 4-10 4-4 10-4-10-10-4 10-4z"/></g>'
        )
    if state == "ecstatic":
        extra += (
            '<path class="heart" d="M196 40c-6-9-20-3-14 8l14 13 14-13c6-11-8-17-14-8z" '
            'fill="#D9614F" stroke="none"/>'
        )
    if mood == "negative":
        # Concerned and supportive: worried brows and a soft frown, no tears.
        eyes += '<path d="M80 104q11-6 21-2 M139 102q10-4 21 2" fill="none"/>'
        mouth = '<path d="M107 154 Q120 145 133 154"/>'
    if state == "very-upset":
        eyes += '<path d="M80 100q11-7 21-2 M139 98q10-5 21 2" fill="none"/>'
    return eyes, mouth, extra


def senti_svg(mood: str = "welcome", confidence: float = 0, labelled: bool = True) -> str:
    """Bare SVG for empty states rendered inline; unlabelled when its parent is labelled."""
    state = mood_state(mood, confidence)
    base = mood if mood in MESSAGES else "idle"
    eyes, mouth, extra = face(base, state)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240" '
        + (f'role="img" aria-label="Senti: {state}">' if labelled else 'aria-hidden="true">')
        + '<ellipse cx="120" cy="221" rx="59" ry="8" fill="#70978933"/>'
        '<g class="body" stroke="#234d42" stroke-width="4" stroke-linecap="round" '
        'stroke-linejoin="round">'
        '<path d="M119 66 Q83 51 96 28 Q126 31 119 66 M120 65Q124 28 151 35Q153 58 120 65" '
        'fill="#65b998"/>'
        + (
            '<path class="wave" d="M174 170q22-14 24-40" fill="none"/>'
            '<path d="M66 178Q40 169 48 152" fill="none"/>'
            if base == "welcome"
            else '<path d="M66 178Q40 169 48 152 M174 178q26-9 18-26" fill="none"/>'
        )
        + '<path d="M91 207l-8 10 M149 207l8 10"/>'
        '<path d="M120 61C66 61 51 109 61 162Q64 210 120 211Q176 210 179 162C189 109 174 61 '
        '120 61Z" fill="#a5ddc4"/><path d="M99 182q21 12 42 0" stroke="#75b99f" fill="none"/>'
        '<g fill="#edaaa0" stroke="none" opacity=".8"><ellipse cx="77" cy="144" rx="12" ry="7"/>'
        '<ellipse cx="163" cy="144" rx="12" ry="7"/></g>'
        f'<g fill="#234d42">{eyes}</g><g fill="none">{mouth}</g>{extra}</g></svg>'
    )


def svg_image(svg: str, label: str = "", size: int = 84) -> str:
    """Inline SVG as an <img> data URI: st.html's sanitizer strips raw <svg> elements."""
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (
        f'<img src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}" '
        f'alt="{html.escape(label)}">'
    )


def senti_img(mood: str = "welcome", size: int = 84) -> str:
    """Static Senti for empty states. Decorative, so it has empty alt text."""
    return svg_image(senti_svg(mood, labelled=False), "", size)


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
    return f"""<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><style>
    :root{{color-scheme:normal;background:transparent}}body{{margin:0;background:transparent;font:15px system-ui,"Segoe UI",sans-serif;text-align:center;color:#1B2B33}}
    .scene{{animation:arrive .3s ease-out;padding:12px 6px}}.bubble{{margin:auto;max-width:300px;padding:12px 16px;border-radius:16px;background:#FFFFFF;color:#1B2B33;border:1px solid #D3DDD6;line-height:1.45;box-shadow:0 2px 8px rgba(27,43,51,.06)}}
    {size}.body{{transform-origin:120px 210px;animation:breathe 3s ease-in-out infinite}}
    .positive .body{{animation:bounce 1.2s ease-in-out infinite}}.ecstatic .body{{animation:bounce .9s ease-in-out infinite}}
    .negative .body,.very-upset .body{{animation:sway 3s ease-in-out infinite}}.thinking .body{{animation:tilt 1.6s ease-in-out infinite}}
    .wave{{transform-box:fill-box;transform-origin:0% 100%;animation:wave 1.4s ease-in-out infinite}}
    .eyes{{transform-box:fill-box;transform-origin:center;animation:blink 4s infinite}}.thinking .eyes{{animation:blink 1.6s infinite}}
    .sparkles{{animation:twinkle 1s infinite alternate}}.heart{{transform-box:fill-box;transform-origin:center;animation:pop 1.2s ease-in-out infinite}}
    .dots circle{{animation:dot 1.2s infinite}}.dots circle:nth-child(2){{animation-delay:.2s}}.dots circle:nth-child(3){{animation-delay:.4s}}
    @keyframes arrive{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:translateY(0)}}}}
    @keyframes breathe{{50%{{transform:translateY(-3px)}}}}@keyframes bounce{{50%{{transform:translateY(-7px)}}}}
    @keyframes sway{{25%{{transform:rotate(-2deg)}}75%{{transform:rotate(2deg)}}}}@keyframes tilt{{50%{{transform:rotate(-3deg)}}}}
    @keyframes wave{{50%{{transform:rotate(-14deg)}}}}@keyframes pop{{50%{{transform:scale(1.15)}}}}@keyframes dot{{0%,100%{{opacity:.25}}50%{{opacity:1}}}}
    @keyframes blink{{0%,92%,100%{{transform:scaleY(1)}}96%{{transform:scaleY(.1)}}}}@keyframes twinkle{{to{{opacity:.3}}}}
    @media(prefers-reduced-motion:reduce){{*{{animation:none!important}}}}
    </style></head><body><div class="scene {state}" role="img" aria-label="Senti: {state}"><div class="bubble">{bubble}</div>
    {senti_svg(mood, confidence, labelled=False)}</div></body></html>"""
