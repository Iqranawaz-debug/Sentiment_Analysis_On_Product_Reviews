"""Shared theme with local system fonts. Page headings are drawn by views.render."""

import streamlit as st


def inject_theme() -> None:
    st.html(
        """<style>
    :root{--ink:#172B3A;--muted:#5B6B78;--line:#DCE3E9;--panel:#F7F9FB;--brand:#137C73;
      --brand-ink:#0E5C55;--pos:#1E7F4F;--neu:#B7791F;--neg:#C0392B}
    .block-container{max-width:1240px;padding-top:1.6rem;padding-bottom:3rem}
    .stMain h2{font-size:1.15rem;font-weight:650;letter-spacing:-.01em;padding-top:.2rem}
    .stMain h3{font-size:1.25rem;font-weight:650;letter-spacing:-.01em}
    .page-head{margin:0 0 1.2rem;padding-bottom:1rem;border-bottom:1px solid var(--line)}
    .page-head h1{font-size:1.9rem;letter-spacing:-.03em;margin:.15rem 0 .2rem;padding:0;color:var(--ink)}
    .page-head p{margin:0;color:var(--muted)}
    .eyebrow{letter-spacing:.14em;font-size:.7rem;text-transform:uppercase;color:var(--brand-ink);font-weight:600}
    .brand{display:flex;gap:.7rem;align-items:center;padding:.2rem 0 .6rem}
    .brand-mark{display:grid;place-items:center;width:2.2rem;height:2.2rem;border-radius:10px;background:var(--brand);color:#fff;font-size:1.1rem}
    .brand strong{display:block;font-size:1.05rem;color:var(--ink)}.brand span{font-size:.78rem;color:var(--muted)}
    .stButton button,.stDownloadButton button{border-radius:10px;min-height:2.6rem}
    .nav-group{font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:700;margin:.9rem 0 .15rem .3rem}
    [class*="st-key-nav_"] button{justify-content:flex-start;min-height:2.3rem;padding:.25rem .7rem;border-radius:8px;font-weight:500}
    [class*="st-key-nav_"] button > div,[class*="st-key-nav_"] button > div > span{justify-content:flex-start;width:100%;gap:.55rem}
    [class*="st-key-nav_"] button[kind="primary"]{box-shadow:inset 3px 0 0 var(--brand-ink);font-weight:650}
    [class*="st-key-nav_"] button[kind="tertiary"]:hover{background:#E6F2EF}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.15rem}
    .chips{display:flex;flex-wrap:wrap;gap:.4rem;justify-content:flex-end;margin-bottom:1.3rem}
    .chip{font-size:.78rem;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:999px;padding:.2rem .65rem;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .onboard{color:var(--muted);font-size:.92rem}.onboard ul{margin:.3rem 0 0 1.1rem;padding:0}.onboard li{margin:.15rem 0}
    .verdict-row{display:flex;flex-wrap:wrap;align-items:center;gap:.6rem;margin:.2rem 0 .4rem}
    .sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
    .stTextArea textarea{border-radius:12px}
    :focus-visible{outline:3px solid #1B6FD1 !important;outline-offset:2px}
    .kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.8rem;margin:.2rem 0 1rem}
    .kpi{background:#fff;border:1px solid var(--line);border-left:4px solid var(--brand);border-radius:12px;padding:.85rem 1rem}
    .kpi.positive{border-left-color:var(--pos)}.kpi.neutral{border-left-color:var(--neu)}.kpi.negative{border-left-color:var(--neg)}
    .kpi-title{font-size:.8rem;color:var(--muted);font-weight:600}
    .kpi-value{font-size:1.65rem;font-weight:700;color:var(--ink);letter-spacing:-.02em;margin:.15rem 0}
    .kpi-note{font-size:.78rem;color:var(--muted);line-height:1.35}
    .context{display:flex;flex-direction:column;gap:.1rem;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:.7rem 1rem}
    .context-label{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
    .context span:last-child{font-size:.85rem;color:var(--muted)}
    .empty{border:1px dashed #B8C4CE;border-radius:14px;padding:1.4rem 1.5rem;background:var(--panel);margin-bottom:1rem}
    .empty h2{margin:0 0 .3rem;font-size:1.15rem}.empty p{margin:0;color:var(--muted);max-width:60ch}
    .insights{list-style:none;margin:0;padding:0;display:grid;gap:.7rem}
    .insights li{border-left:3px solid var(--brand);padding:.1rem 0 .1rem .75rem}
    .insights strong{display:block;color:var(--ink);font-size:.92rem}.insights span{color:var(--muted);font-size:.86rem;line-height:1.45}
    .facts{display:grid;grid-template-columns:minmax(120px,190px) 1fr;gap:.45rem 1rem;margin:0}
    .facts dt{font-weight:600;color:var(--ink);font-size:.88rem}.facts dd{margin:0;color:var(--muted);font-size:.88rem}
    .muted{color:var(--muted);font-size:.9rem}
    .verdict{display:inline-block;border-radius:30px;background:#E6F2EF;color:var(--ink);padding:.45rem 1rem;font-weight:700;border:1px solid transparent}
    .verdict.positive{background:#E3F3EA;color:#14593A;border-color:#A9D8BD}
    .verdict.neutral{background:#FBF1DF;color:#7A4F12;border-color:#E9C98F}
    .verdict.negative{background:#FBE6E3;color:#8E2A20;border-color:#EDB3AB}
    .explanation{line-height:2.2;padding:1rem;border:1px solid #80a79866;border-radius:16px}
    /* The note keeps its space while hidden: if it collapsed on blur, the button below
       would jump during the mouse press and the user's first click would miss. */
    .editing-review-note{visibility:hidden;font-size:.85rem;color:inherit;min-height:1.3em}
    .stMain:has(textarea:focus) .st-key-live_verdict,
    .stMain:has(textarea:focus) .st-key-live_mascot{display:none}
    .stMain:has(textarea:focus) .editing-review-note{visibility:visible}
    @media(max-width:640px){.page-head h1{font-size:1.5rem}.block-container{padding-left:1rem;padding-right:1rem}
      .facts{grid-template-columns:1fr}.facts dd{margin-bottom:.4rem}.kpi-value{font-size:1.4rem}}
    @media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
    </style>""",
    )
