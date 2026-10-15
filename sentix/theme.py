"""Shared theme with local system fonts."""

import streamlit as st


def inject_theme() -> None:
    st.html(
        """<style>
    .block-container{max-width:1200px;padding-top:2rem}
    .sentix-header{background:linear-gradient(120deg,#173e38,#34735e);color:#f0fff8;padding:2rem;border-radius:26px;margin-bottom:1.7rem;box-shadow:0 10px 30px #173e3818}
    .sentix-header h1{color:#f0fff8;margin:0;font-size:2.7rem;letter-spacing:-.06em}
    .sentix-header p{margin:.4rem 0 0;opacity:.85}.eyebrow{letter-spacing:.18em;font-size:.73rem}
    .stButton button{border-radius:12px}.stTextArea textarea{border-radius:14px}
    [data-baseweb="tab-list"]{gap:1rem;flex-wrap:wrap}
    .verdict{display:inline-block;border-radius:30px;background:#d9eee4;color:#173e38;padding:.5rem 1rem;font-weight:700}
    .explanation{line-height:2.2;padding:1rem;border:1px solid #80a79866;border-radius:16px}
    .editing-review-note{display:none}
    .stMain:has(textarea:focus) .st-key-live_verdict,
    .stMain:has(textarea:focus) .st-key-live_mascot{display:none}
    .stMain:has(textarea:focus) .editing-review-note{display:block;font-size:.85rem;color:inherit}
    @media(max-width:640px){.sentix-header{padding:1.3rem}.sentix-header h1{font-size:2.1rem}}
    </style><header class="sentix-header"><span class="eyebrow">REAL REVIEWS. REAL SIGNALS. A LITTLE PERSONALITY.</span>
    <h1>Sentix AI ✦</h1><p>Meet Senti. The little face behind your customer feedback.</p></header>""",
    )
