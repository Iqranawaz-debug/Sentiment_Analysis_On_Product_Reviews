"""
Sentix AI - Mascot Demo
Run:  pip install streamlit vaderSentiment
      streamlit run sentix_mascot_app.py

Replace predict_sentiment() with YOUR trained model's prediction
(it must return: label, confidence) and the mascot will react to it.
"""
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Sentix AI", page_icon="🐱", layout="centered")


# ------------------------------------------------------------------
# 1) YOUR MODEL GOES HERE
# ------------------------------------------------------------------
def predict_sentiment(text: str):
    """Return (label, confidence) where label in {'positive','neutral','negative'}.
    Demo uses VADER. Swap with: model.predict_proba(vectorizer.transform([text]))"""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        score = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
    except ImportError:  # tiny fallback so the demo still runs
        good = {"love", "great", "amazing", "perfect", "excellent", "good"}
        bad = {"bad", "worst", "broke", "terrible", "waste", "hate", "poor"}
        words = set(text.lower().split())
        score = 0.6 * (len(words & good) - len(words & bad))
    if score >= 0.25:
        return "positive", min(1.0, 0.5 + score / 2)
    if score <= -0.25:
        return "negative", min(1.0, 0.5 + abs(score) / 2)
    return "neutral", 1.0 - abs(score)


# ------------------------------------------------------------------
# 2) ORIGINAL MASCOT "SENTI" (pure SVG + CSS animation, no images needed)
# ------------------------------------------------------------------
MOODS = {
    "positive": {
        "anim": "bounce", "bubble": "Yay! Customers love this! 🎉", "color": "#4FC3F7",
        "eyes": '<path d="M70 92 Q80 78 90 92" stroke="#222" stroke-width="5" fill="none" stroke-linecap="round"/>'
                '<path d="M110 92 Q120 78 130 92" stroke="#222" stroke-width="5" fill="none" stroke-linecap="round"/>',
        "mouth": '<path d="M78 112 Q100 140 122 112 Z" fill="#E53935" stroke="#222" stroke-width="3"/>',
        "extra": '<circle cx="62" cy="108" r="9" fill="#FF8A80" opacity=".7"/><circle cx="138" cy="108" r="9" fill="#FF8A80" opacity=".7"/>'
                 '<text x="150" y="55" font-size="22" class="spark">✨</text><text x="30" y="60" font-size="18" class="spark">✨</text>',
    },
    "negative": {
        "anim": "sway", "bubble": "Oh no... they are upset 😢", "color": "#90A4AE",
        "eyes": '<ellipse cx="80" cy="92" rx="8" ry="10" fill="#222"/><ellipse cx="120" cy="92" rx="8" ry="10" fill="#222"/>'
                '<circle cx="83" cy="88" r="3" fill="#fff"/><circle cx="123" cy="88" r="3" fill="#fff"/>'
                '<path d="M65 78 L90 84" stroke="#222" stroke-width="4" stroke-linecap="round"/>'
                '<path d="M135 78 L110 84" stroke="#222" stroke-width="4" stroke-linecap="round"/>',
        "mouth": '<path d="M82 128 Q100 108 118 128" stroke="#222" stroke-width="5" fill="none" stroke-linecap="round"/>',
        "extra": '<path d="M72 104 Q68 118 74 122 Q80 118 72 104Z" fill="#4FC3F7" class="tear"/>',
    },
    "neutral": {
        "anim": "blink", "bubble": "Hmm... just okay, nothing special 😐", "color": "#AED581",
        "eyes": '<circle cx="80" cy="92" r="8" fill="#222" class="eye"/><circle cx="120" cy="92" r="8" fill="#222" class="eye"/>',
        "mouth": '<line x1="84" y1="122" x2="116" y2="122" stroke="#222" stroke-width="5" stroke-linecap="round"/>',
        "extra": "",
    },
}


def mascot_html(mood: str) -> str:
    m = MOODS[mood]
    return f"""
    <style>
      body {{ margin:0; font-family: 'Segoe UI', sans-serif; text-align:center; background:transparent; }}
      .bubble {{ display:inline-block; background:#fff; border:3px solid #222; border-radius:18px;
                 padding:8px 16px; font-weight:700; margin-bottom:10px; color:#222; }}
      svg {{ width:230px; height:230px; }}
      .bounce {{ animation: bounce .7s infinite alternate ease-in-out; transform-origin:bottom; }}
      .sway   {{ animation: sway 1.6s infinite ease-in-out; transform-origin:bottom; }}
      .blink .eye {{ animation: blink 3s infinite; transform-origin:center; transform-box:fill-box; }}
      .spark {{ animation: pop 1s infinite alternate; }}
      .tear  {{ animation: drip 1.4s infinite; }}
      @keyframes bounce {{ from {{ transform: translateY(0) }} to {{ transform: translateY(-18px) }} }}
      @keyframes sway   {{ 0%,100% {{ transform: rotate(-3deg) }} 50% {{ transform: rotate(3deg) }} }}
      @keyframes blink  {{ 0%,92%,100% {{ transform: scaleY(1) }} 96% {{ transform: scaleY(.1) }} }}
      @keyframes pop    {{ from {{ opacity:.3 }} to {{ opacity:1 }} }}
      @keyframes drip   {{ 0% {{ transform: translateY(0); opacity:1 }} 100% {{ transform: translateY(26px); opacity:0 }} }}
    </style>
    <div class="bubble">{m['bubble']}</div><br>
    <svg viewBox="0 0 200 220" class="{m['anim']}">
      <ellipse cx="100" cy="205" rx="55" ry="8" fill="#00000022"/>
      <circle cx="42" cy="48" r="16" fill="{m['color']}" stroke="#222" stroke-width="3"/>
      <circle cx="158" cy="48" r="16" fill="{m['color']}" stroke="#222" stroke-width="3"/>
      <ellipse cx="100" cy="170" rx="42" ry="38" fill="{m['color']}" stroke="#222" stroke-width="3"/>
      <ellipse cx="100" cy="178" rx="26" ry="24" fill="#fff" stroke="#222" stroke-width="2"/>
      <circle cx="100" cy="100" r="62" fill="{m['color']}" stroke="#222" stroke-width="3"/>
      <ellipse cx="100" cy="108" rx="48" ry="42" fill="#fff"/>
      {m['eyes']}
      <circle cx="100" cy="104" r="6" fill="#E53935"/>
      {m['mouth']}
      {m['extra']}
    </svg>"""


# ------------------------------------------------------------------
# 3) UI
# ------------------------------------------------------------------
st.title("🐱 Sentix AI")
st.caption("Meet **Senti** - the mascot who reads product reviews and reacts with feelings.")

samples = {
    "😍 Love it": "Absolutely love this product, works perfectly and arrived early!",
    "😡 Hate it": "Terrible quality. It broke in two days. Total waste of money.",
    "😐 Meh": "It is okay. Does the job, nothing special.",
}
cols = st.columns(3)
for col, (label, txt) in zip(cols, samples.items()):
    if col.button(label, use_container_width=True):
        st.session_state["review"] = txt

review = st.text_area("Paste a product review:", key="review", height=120)

if review.strip():
    label, conf = predict_sentiment(review)
    components.html(mascot_html(label), height=340)
    st.subheader(f"Verdict: {label.upper()}")
    st.progress(float(conf), text=f"Confidence: {conf:.0%}")
else:
    components.html(mascot_html("neutral"), height=340)
    st.info("Type or pick a review and watch Senti react 👆")
