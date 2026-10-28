"""Sentix design system: tokens and shared component styles. Page headings are drawn by
views.render. Colours were contrast-checked: every text pair meets WCAG AA (4.5:1);
coral and amber accents are decorative only."""

import streamlit as st

TOKENS = """
:root{
  --bg:#FBFAF6;--surface:#FFFFFF;--mint:#EEF6F1;--mint-2:#E2F0E8;--line:#E3E8E2;--line-2:#D3DDD6;
  --ink:#1B2B33;--muted:#56656F;--brand:#0E6B61;--brand-ink:#0A5149;--brand-soft:#DCEFE8;
  --coral:#D9614F;--amber:#E0A63A;--focus:#1B6FD1;
  --pos:#1E7A4C;--neu:#8A5A0B;--neg:#B5382B;
  --pos-bg:#E2F3E9;--pos-fg:#14593A;--pos-line:#A9D8BD;
  --neu-bg:#FBF0DC;--neu-fg:#7A4F12;--neu-line:#E9C98F;
  --neg-bg:#FBE5E1;--neg-fg:#8E2A20;--neg-line:#EDB3AB;
  --r-sm:8px;--r-md:12px;--r-lg:14px;--r-pill:999px;
  --shadow:0 1px 2px rgba(27,43,51,.04),0 2px 8px rgba(27,43,51,.05);
  --shadow-hover:0 2px 4px rgba(27,43,51,.05),0 8px 20px rgba(27,43,51,.08);
  --ease:cubic-bezier(.2,.7,.3,1);--fast:.15s;--med:.25s;
}"""

STYLES = """
/* Layout and type scale */
.stApp{background:var(--bg)}
.block-container{max-width:1240px;padding-top:1.4rem;padding-bottom:3rem}
.stMain h2{font-size:1.15rem;font-weight:650;letter-spacing:-.01em;padding-top:.2rem;color:var(--ink)}
.stMain h3{font-size:1.2rem;font-weight:650;letter-spacing:-.01em;color:var(--ink)}
.stMain p,.stMain li{line-height:1.55}
.page-head{margin:0 0 1.2rem;padding-bottom:1rem;border-bottom:1px solid var(--line)}
.page-head h1{font-size:1.9rem;letter-spacing:-.03em;margin:.15rem 0 .25rem;padding:0;color:var(--ink)}
.page-head p{margin:0;color:var(--muted);font-size:1rem}
.eyebrow{letter-spacing:.14em;font-size:.7rem;text-transform:uppercase;color:var(--brand);font-weight:700}
.muted{color:var(--muted);font-size:.9rem}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}

/* Cards: Streamlit bordered containers share one surface style */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]){border-radius:var(--r-lg)}
.stMain [data-testid="stVerticalBlockBorderWrapper"]{background:var(--surface);border-color:var(--line)!important;
  box-shadow:var(--shadow);transition:box-shadow var(--med) var(--ease),transform var(--med) var(--ease)}
.stMain [data-testid="stVerticalBlockBorderWrapper"]:hover{box-shadow:var(--shadow-hover)}
.stMain [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"]{box-shadow:none}

/* Sidebar */
[data-testid="stSidebar"]{background:var(--mint);border-right:1px solid var(--line)}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.15rem}
.brand{display:flex;gap:.7rem;align-items:center;padding:.15rem 0 .5rem}
.brand-mark{display:grid;place-items:center;width:2.4rem;height:2.4rem;border-radius:12px;background:var(--brand);
  box-shadow:inset 0 -3px 0 rgba(0,0,0,.12)}
.brand-mark img{display:block}
.brand strong{display:block;font-size:1.08rem;color:var(--ink);letter-spacing:-.01em}
.brand span{font-size:.78rem;color:var(--muted)}
.nav-group{font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:700;margin:.95rem 0 .2rem .35rem}
[class*="st-key-nav_"] button{justify-content:flex-start;min-height:2.35rem;padding:.25rem .7rem;border-radius:var(--r-sm);
  font-weight:500;color:var(--ink);transition:background var(--fast) var(--ease)}
[class*="st-key-nav_"] button > div,[class*="st-key-nav_"] button > div > span{justify-content:flex-start;width:100%;gap:.6rem}
[class*="st-key-nav_"] button[kind="tertiary"]:hover{background:var(--mint-2);color:var(--ink)}
[class*="st-key-nav_"] button[kind="primary"]{background:var(--brand-soft);color:var(--brand-ink);border:1px solid #BFDDD2;
  box-shadow:inset 3px 0 0 var(--brand);font-weight:650}
[class*="st-key-nav_"] button[kind="primary"]:hover{background:#D0E9E0}
.side-label{font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:700;margin:.2rem 0 .1rem .35rem}

/* Buttons and inputs */
.stButton button,.stDownloadButton button,[data-testid="stPopover"] button{border-radius:10px;min-height:2.6rem;
  transition:background var(--fast) var(--ease),box-shadow var(--fast) var(--ease),transform var(--fast) var(--ease)}
.stMain .stButton button:not(:disabled):hover,.stMain .stDownloadButton button:not(:disabled):hover{box-shadow:var(--shadow-hover)}
.stMain .stButton button:not(:disabled):active{transform:translateY(1px)}
.stButton button:disabled{opacity:.55;cursor:not-allowed}
.stTextArea textarea,.stTextInput input{border-radius:var(--r-md);background:var(--surface)}
.stTextArea textarea:focus,.stTextInput input:focus{box-shadow:0 0 0 3px rgba(14,107,97,.18)}
[data-testid="InputInstructions"]{color:var(--muted)!important;font-size:.78rem}
[class*="st-key-sample_"] button{border-radius:var(--r-pill);min-height:2.2rem;background:var(--mint);border-color:var(--line-2);font-size:.9rem}
[class*="st-key-sample_"] button:hover{background:var(--mint-2);border-color:var(--brand)}
:focus-visible{outline:3px solid var(--focus)!important;outline-offset:2px}

/* Header chips */
.chips{display:flex;flex-wrap:wrap;gap:.4rem;justify-content:flex-end;margin-bottom:1.3rem}
.chip{font-size:.78rem;border:1px solid var(--line-2);background:var(--surface);color:var(--ink);border-radius:var(--r-pill);
  padding:.22rem .7rem;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chip b{color:var(--brand-ink);font-weight:650}

/* Result card */
.onboard{color:var(--muted);font-size:.93rem}.onboard ul{margin:.35rem 0 0 1.1rem;padding:0}.onboard li{margin:.18rem 0}
.verdict-row{display:flex;flex-wrap:wrap;align-items:center;gap:.6rem;margin:.1rem 0 .2rem;animation:rise var(--med) var(--ease)}
.verdict{display:inline-flex;align-items:center;gap:.35rem;border-radius:var(--r-pill);padding:.42rem 1rem;font-weight:700;
  border:1px solid transparent;font-size:1rem}
.verdict.positive{background:var(--pos-bg);color:var(--pos-fg);border-color:var(--pos-line)}
.verdict.neutral{background:var(--neu-bg);color:var(--neu-fg);border-color:var(--neu-line)}
.verdict.negative{background:var(--neg-bg);color:var(--neg-fg);border-color:var(--neg-line)}
.reading{margin:.1rem 0 .3rem;color:var(--ink);font-size:.98rem;animation:rise var(--med) var(--ease)}
.reading .tag{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  border-radius:var(--r-pill);padding:.1rem .5rem;margin-right:.4rem;vertical-align:1px}
.reading .tag.clear{background:var(--brand-soft);color:var(--brand-ink)}
.reading .tag.tentative{background:var(--neu-bg);color:var(--neu-fg)}
.result-foot{border-top:1px solid var(--line);margin-top:.4rem;padding-top:.5rem;color:var(--muted);font-size:.8rem}
.editing-result-note{display:none;color:var(--muted);font-size:.9rem;padding:.4rem 0}
/* While the review is being edited, the previous result is hidden so it can never be read
   as belonging to the new text. The card itself stays, so the layout does not jump. */
.stMain:has(textarea:focus) .st-key-live_result,
.stMain:has(textarea:focus) .st-key-live_mascot{display:none}
.stMain:has(textarea:focus) .editing-result-note{display:block}
/* The note keeps its space while hidden: if it collapsed on blur, the button below
   would jump during the mouse press and the user's first click would miss. */
.editing-review-note{visibility:hidden;font-size:.85rem;color:var(--muted);min-height:1.3em}
.stMain:has(textarea:focus) .editing-review-note{visibility:visible}

/* Dashboard pieces */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.8rem;margin:.2rem 0 1rem}
.kpi{background:var(--surface);border:1px solid var(--line);border-top:3px solid var(--brand);border-radius:var(--r-md);
  padding:.85rem 1rem;box-shadow:var(--shadow);transition:box-shadow var(--med) var(--ease),transform var(--med) var(--ease)}
.kpi:hover{box-shadow:var(--shadow-hover);transform:translateY(-1px)}
.kpi.positive{border-top-color:var(--pos)}.kpi.neutral{border-top-color:var(--amber)}.kpi.negative{border-top-color:var(--neg)}
.kpi-title{font-size:.8rem;color:var(--muted);font-weight:650}
.kpi-value{font-size:1.65rem;font-weight:700;color:var(--ink);letter-spacing:-.02em;margin:.15rem 0;font-variant-numeric:tabular-nums}
.kpi-note{font-size:.78rem;color:var(--muted);line-height:1.35}
.context{display:flex;flex-direction:column;gap:.1rem;background:var(--mint);border:1px solid var(--line);border-radius:var(--r-md);padding:.7rem 1rem}
.context-label{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:700}
.context span:last-child{font-size:.85rem;color:var(--muted)}
.empty{display:flex;gap:1.2rem;align-items:center;border:1px dashed var(--line-2);border-radius:var(--r-lg);padding:1.2rem 1.4rem;
  background:var(--mint);margin-bottom:1rem}
.empty h2{margin:0 0 .3rem;font-size:1.15rem}.empty p{margin:0;color:var(--muted);max-width:60ch}
.empty img{flex:none}
.insights{list-style:none;margin:0;padding:0;display:grid;gap:.75rem}
.insights li{border-left:3px solid var(--brand);padding:.1rem 0 .1rem .75rem}
.insights strong{display:block;color:var(--ink);font-size:.93rem}.insights span{color:var(--muted);font-size:.86rem;line-height:1.45}
.facts{display:grid;grid-template-columns:minmax(120px,190px) 1fr;gap:.45rem 1rem;margin:0}
.facts dt{font-weight:650;color:var(--ink);font-size:.88rem}.facts dd{margin:0;color:var(--muted);font-size:.88rem}
.explanation{line-height:2.2;padding:1rem;border:1px solid var(--line);border-radius:var(--r-lg);background:var(--surface)}
.legend{display:flex;flex-wrap:wrap;gap:1rem;font-size:.85rem;color:var(--muted);margin:.4rem 0}
.legend i{display:inline-block;width:.9rem;height:.9rem;border-radius:4px;margin-right:.35rem;vertical-align:-2px}
.scoreboard{display:flex;gap:.8rem;align-items:stretch;margin:.3rem 0 .8rem}
.score{flex:1;background:var(--mint);border:1px solid var(--line);border-radius:var(--r-md);padding:.6rem .9rem;text-align:center}
.score b{display:block;font-size:1.6rem;color:var(--ink);font-variant-numeric:tabular-nums}
.score span{font-size:.8rem;color:var(--muted)}
[data-testid="stDataFrame"]{border-radius:var(--r-md);overflow:hidden}
[data-testid="stAlert"]{border-radius:var(--r-md)}

@keyframes rise{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
@media(max-width:640px){.page-head h1{font-size:1.5rem}.block-container{padding-left:1rem;padding-right:1rem}
  .facts{grid-template-columns:1fr}.facts dd{margin-bottom:.4rem}.kpi-value{font-size:1.4rem}
  .chips{justify-content:flex-start}.empty{flex-direction:column;align-items:flex-start}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""


def inject_theme() -> None:
    st.html(f"<style>{TOKENS}{STYLES}</style>")
