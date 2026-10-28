"""Live review and saved evaluation views."""

import html
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sentix import charts, dashboard
from sentix.dataset import badge
from sentix.deep import DeepPredictor
from sentix.mascot import presentation_mood, render_senti, unsure_threshold
from sentix.state import go, initial, keep_widget_state, slug
from src.ui import load_reports

PAGES = {
    "Overview": "Sentiment, certainty and insights across the active dataset.",
    "Review Analyzer": "Analyze one review with the real model and see how sure it is.",
    "Review Explorer": "Search, filter and open individual reviews in the active dataset.",
    "Batch Analysis": "Score a CSV of reviews, map optional columns and download results.",
    "History": "Reviews you analyzed in this browser session.",
    "Explain": "Which words pushed the Fast model toward positive or negative.",
    "Model & Evaluation": "What the model is, how it was measured and where it struggles.",
    "Guess the Mood": "A practice game: guess a real review's star-rating sentiment.",
}
TITLES = {"Overview": "Customer Intelligence Overview"}
NAV = {
    "Analyze": ["Overview", "Review Analyzer", "Review Explorer", "Batch Analysis", "History"],
    "Model": ["Explain", "Model & Evaluation"],
    "Practice": ["Guess the Mood"],
}
ICONS = {
    "Overview": ":material/dashboard:",
    "Review Analyzer": ":material/rate_review:",
    "Review Explorer": ":material/manage_search:",
    "Batch Analysis": ":material/upload_file:",
    "History": ":material/history:",
    "Explain": ":material/lightbulb:",
    "Model & Evaluation": ":material/analytics:",
    "Guess the Mood": ":material/sports_esports:",
}
DEFAULT_PAGE = "Review Analyzer"

EXAMPLES = {
    "Praise": "Absolutely love this product, works perfectly and arrived early!",
    "Complaint": "Terrible quality. It broke in two days. Total waste of money.",
    "Lukewarm": "It is okay. Does the job, nothing special.",
}


def load_example(text: str) -> None:
    """Examples only fill the text box; analysis always needs an explicit click."""
    st.session_state.review = text
    clear_review_result()


def clear_review() -> None:
    st.session_state.review = ""
    clear_review_result()
    st.session_state.pop("analyzed_text", None)


def model_version(predictor, result: dict) -> str:
    """What actually produced this result, kept with history entries."""
    if result.get("route") == "transformer":
        return "int8 CardiffNLP Twitter-RoBERTa (Deep Mode)"
    sha = predictor.metadata.get("artifact_sha256", "unknown")[:8]
    rule = result.get("decision_rule", "most probable class")
    return f"TF-IDF logistic regression {sha} · {rule}"


def save_feedback(text: str, expected: str) -> None:
    """Attach the user's correction to this session's history entry for the review."""
    for entry in reversed(st.session_state.get("history", [])):
        if entry["review_text"] == text:
            entry["feedback"] = f"Reported incorrect; expected {expected}"
            break
    st.toast("Thanks. Feedback saved to this session's history.")


def clear_review_result() -> None:
    """Discard the old prediction as soon as Streamlit receives an edit."""
    st.session_state.result = None


def clear_mode_results() -> None:
    clear_review_result()
    for key in ("analyzed_text", "bulk_result", "bulk_signature", "explanation"):
        st.session_state.pop(key, None)
    st.session_state.game_reveal = None
    st.session_state.game_score = [0, 0]


def live_review(predictor) -> None:
    mode = st.session_state.get("analysis_mode", "Fast Mode")
    left, right = st.columns([1.05, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("Review")
        text = st.text_area(
            "Product review",
            key="review",
            height=190,
            max_chars=20000,
            on_change=clear_review_result,
            placeholder="Paste or type a customer review, then press Analyze review.",
            help="1 to 20,000 characters. English product reviews work best.",
        )
        st.html(
            '<div class="editing-review-note">Editing review. Press Analyze review when ready.</div>'
        )
        a, b = st.columns([2, 1])
        requested = a.button("Analyze review", type="primary", width="stretch")
        b.button("Clear", on_click=clear_review, width="stretch")
        st.caption("Load an example review (written for this demo; it only fills the text box):")
        for col, (label, sample) in zip(st.columns(3), EXAMPLES.items()):
            col.button(label, on_click=load_example, args=(sample,), width="stretch")
        st.caption(
            f"Mode: **{mode}**. "
            + (
                "The TF-IDF model reads every review."
                if mode == "Fast Mode"
                else "Reviews the TF-IDF model is less than 80% sure of go to a transformer."
            )
            + " Change the mode in the sidebar."
        )
        if requested:
            try:
                with st.spinner("Analyzing…"):
                    started = time.perf_counter()
                    st.session_state.result = predictor.predict(text)
                    elapsed = (time.perf_counter() - started) * 1000
                st.session_state.analyzed_text = text
                st.session_state.result_meta = {
                    "elapsed_ms": elapsed,
                    "at": time.strftime("%H:%M:%S UTC", time.gmtime()),
                    "mode": mode,
                    "model": model_version(predictor, st.session_state.result),
                }
                if st.session_state.result["status"] == "ok":
                    dashboard.record_history(
                        text,
                        st.session_state.result,
                        mode,
                        elapsed,
                        st.session_state.result_meta["model"],
                    )
            except Exception:
                st.session_state.result = None
                st.session_state.analyzed_text = None
                st.session_state.result_error = True
        result = st.session_state.get("result")
        if text != st.session_state.get("analyzed_text"):
            result = None
            st.session_state.result = None
    with right:
        with st.container(border=True, key="live_verdict"):
            st.subheader("Result")
            if st.session_state.pop("result_error", False):
                st.error(
                    "The model could not analyze this review. Your text is kept; try again "
                    "or shorten it. No sentiment was guessed."
                )
            elif text.strip() and st.session_state.get("analyzed_text") not in (None, text):
                st.info("The review changed since the last analysis. Press Analyze review.")
            show_result(text, result)
        if st.session_state.get("show_senti", True):
            valid = text.strip() and result and result["status"] == "ok"
            with st.container(key="live_mascot"):
                st.iframe(
                    render_senti(
                        presentation_mood(result) if valid else "idle",
                        result["confidence"] if valid else 0,
                        compact=True,
                    ),
                    height=230,
                )


def show_result(text: str, result: dict | None) -> None:
    """Keep model probabilities intact while making uncertainty explicit."""
    if not (text.strip() and result):
        st.html(
            '<div class="onboard"><p>Results appear here after you press '
            "<strong>Analyze review</strong>:</p><ul><li>the predicted sentiment</li>"
            "<li>calibrated probabilities for all three classes</li><li>how the label was "
            "chosen and any reason to treat it as tentative</li><li>the model, mode and "
            "time taken</li></ul></div>"
        )
        return
    if result["status"] != "ok":
        st.warning(
            "Not analyzed: "
            + result.get("message", "the model needs a little more text.")
            + " No sentiment is guessed for text the model cannot read."
        )
        st.html('<div role="status" aria-live="polite" class="sr-only">Review not analyzed.</div>')
        return
    label = result["label"]
    tentative = presentation_mood(result) == "unsure"
    if result.get("warning"):
        st.warning(result["warning"])
    st.html(
        f'<div class="verdict-row"><span class="verdict {label}">{html.escape(badge(label))}'
        f'</span><span class="muted">{"Tentative" if tentative else "Clear"} result · '
        f"probability {result['confidence']:.1%}</span></div>"
        f'<div role="status" aria-live="polite" class="sr-only">Result: {html.escape(label)}, '
        f"probability {result['confidence']:.0%}{', tentative' if tentative else ''}.</div>"
    )
    st.altair_chart(charts.probabilities(result["probabilities"]), width="stretch")
    summary = ", ".join(f"{k} {v:.1%}" for k, v in result["probabilities"].items())
    st.caption(f"Probabilities: {summary}. They describe model uncertainty, not correctness.")
    for note in uncertainty_notes(result):
        st.warning(note)
    if tentative and not result.get("uncertainty"):
        st.info(
            f"Tentative: the probability is below {unsure_threshold():.0%}, so read the "
            "review yourself before relying on this label."
        )
    meta = st.session_state.get("result_meta") or {}
    if meta:
        st.caption(
            f"{meta['mode']} · {meta.get('model', '')} · {meta['elapsed_ms']:.0f} ms · {meta['at']}"
        )
    copy, report = st.columns(2)
    with copy.popover("Copy result", icon=":material/content_copy:", width="stretch"):
        st.code(
            f"Review: {text}\nSentiment: {label} ({result['confidence']:.1%})\n"
            f"Probabilities: {summary}\nModel: {meta.get('model', '')}\n"
            f"Mode: {meta.get('mode', '')}",
            language=None,
            wrap_lines=True,
        )
        st.caption("Use the copy button in the box's corner.")
    with report.popover("Report incorrect", icon=":material/flag:", width="stretch"):
        expected = st.radio(
            "What should the sentiment be?",
            [x for x in ("negative", "neutral", "positive") if x != label] + ["not sure"],
            key="feedback_expected",
        )
        if st.button("Save feedback", key="save_feedback"):
            save_feedback(text, expected)
        st.caption("Saved with this session's history only; nothing is sent anywhere.")


def uncertainty_notes(result: dict) -> list[str]:
    """Plain-language reasons a label is tentative. They never change the label."""
    notes = []
    if "rule_changed_label" in result.get("uncertainty", []):
        notes.append(
            f"The most probable class is {result['top_label']} "
            f"({result['top_probability']:.1%}), but the probabilities lean positive because "
            "78% of training reviews were positive. After correcting for that skew with the "
            f"validated decision rule, the label is {result['label']}. Treat it as tentative."
        )
    if "classifier_disagrees" in result.get("uncertainty", []):
        notes.append(
            f"Before calibration the classifier ranked {result['raw_label']} first, but the "
            f"final label is {result['label']}. The model is split on this review; read it "
            "yourself."
        )
    return notes


@st.cache_data
def read_metrics(directory: str, metadata: dict, modified: int) -> dict:
    return load_reports(Path(directory), metadata)


def model_lab(predictor) -> None:
    st.subheader("The brain behind the face")
    st.write(
        "TF-IDF features → calibrated classifier → learned negative / neutral / positive labels."
    )
    directory = Path(
        os.environ.get("SENTIMENT_REPORTS_DIR", Path(__file__).resolve().parents[1] / "reports")
    )
    try:
        report = read_metrics(
            str(directory), predictor.metadata, (directory / "metrics.json").stat().st_mtime_ns
        )
    except Exception:
        report = None
    model_info(predictor, report)
    if report is None:
        st.info(
            "Senti's matching evaluation reports are unavailable. Run the notebook/evaluation workflow to generate metrics; no scores are invented here."
        )
        return
    decision_section(predictor, directory)
    mode_comparison(directory)
    try:
        evaluation(report, directory)
    except Exception:
        st.info("Some saved evaluation details could not be read. No scores are invented here.")


DEEP_RESULTS = "deep_mode_e2e_20261028_batch1/results.json"


def mode_comparison(directory: Path) -> None:
    """Fast vs Deep on the same held-out reviews, from saved end-to-end runs only."""
    with st.container(border=True):
        st.header("Fast vs Deep Mode")
        try:
            fast = json.loads((directory / "decision_policy.json").read_text(encoding="utf-8"))
            deep = json.loads((directory / DEEP_RESULTS).read_text(encoding="utf-8"))
            assert deep["acceptance"]["batch_independent"]
        except OSError, ValueError, KeyError, AssertionError:
            st.info(
                "No matching Deep Mode end-to-end run is saved. Run python -m "
                "scripts.verify_deep_mode to measure it; nothing is estimated here."
            )
            return
        rows = []
        for name, test, regression in (
            ("Fast (decision rule)", fast["test"]["app"], fast["regression"]["app"]),
            ("Deep hybrid", deep["test"], deep["regression"]),
        ):
            matrix = np.array(test["confusion_matrix"])[:3, :3]
            recall = np.diag(matrix) / matrix.sum(axis=1)
            rows.append(
                {
                    "Mode": name,
                    "Reviews": test["n"],
                    "Accuracy": test["accuracy"],
                    "Macro F1": test["macro_f1"],
                    "Negative recall": recall[0],
                    "Neutral recall": recall[1],
                    "Positive recall": recall[2],
                    "Regression correct": f"{round(regression['accuracy'] * regression['n'])}"
                    f"/{regression['n']}",
                }
            )
        percent = st.column_config.NumberColumn(format="percent")
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "Accuracy": percent,
                "Negative recall": percent,
                "Neutral recall": percent,
                "Positive recall": percent,
                "Macro F1": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        seconds = deep["test"]["seconds"] / max(deep["test"]["transformer_rows"], 1)
        st.caption(
            f"Same held-out reviews. Deep sent {deep['test']['transformer_rows']:,} reviews "
            f"(below 0.80 Fast probability) to the int8 transformer at about "
            f"{seconds * 1000:.0f} ms each on CPU, scoring one review at a time so results "
            "do not depend on other reviews. Deep finds more neutral reviews but misses more "
            "negative ones; it is not better overall. Both rows come from the app's own "
            "predictors, with unreadable reviews counted as wrong."
        )


def decision_section(predictor, directory: Path) -> None:
    """Before/after evaluation of the decision rule from reports/decision_policy.json."""
    path = directory / "decision_policy.json"
    with st.container(border=True):
        st.header("Decision rule evaluation")
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["artifact_sha256"] != predictor.metadata["artifact_sha256"]:
                raise ValueError
        except OSError, ValueError, KeyError:
            st.info(
                "No decision-rule evaluation matches this model. Labels are the most probable "
                "class. Run python -m scripts.tune_decision_policy to evaluate the rule."
            )
            return
        active = predictor.policy is not None
        st.caption(
            f"Labels are chosen by {predictor.decision_rule}. "
            + (
                "Calibrated probabilities are unchanged; only the label choice corrects for "
                "the training data's 78% positive skew."
                if active
                else "The rule is not active for this model."
            )
        )
        test, regression = report["test"], report["regression"]
        rows = []
        for name, block in (
            ("Held-out test", test),
            ("Short reviews (≤8 words)", test["short_reviews"]),
            ("Curated regression cases", regression),
        ):
            for stage in ("before", "after"):
                s = block[stage]
                rows.append(
                    {
                        "Set": name,
                        "Decision": "Most probable class"
                        if stage == "before"
                        else "Class-prior rule",
                        "Reviews": s["n"],
                        "Accuracy": s["accuracy"],
                        "Macro F1": s["macro_f1"],
                        **{
                            f"{k.title()} recall": s["per_class"][k]["recall"]
                            for k in ("negative", "neutral", "positive")
                        },
                    }
                )
        percent = st.column_config.NumberColumn(format="percent")
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "Accuracy": percent,
                "Negative recall": percent,
                "Neutral recall": percent,
                "Positive recall": percent,
                "Macro F1": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        delta = test["paired_bootstrap_delta"]
        st.caption(
            f"Alpha {report['alpha']:g} was chosen on the validation split only; the test "
            "split and regression cases were scored once afterwards. "
            f"Paired 95% bootstrap change on test: accuracy {delta['accuracy']['lower']:+.1%} to "
            f"{delta['accuracy']['upper']:+.1%}, macro F1 {delta['macro_f1']['lower']:+.3f} to "
            f"{delta['macro_f1']['upper']:+.3f}. Short reviews got slightly worse. Both splits "
            "exclude the production model's training sample; labels are star-rating proxies."
        )
        cases = pd.DataFrame(regression["cases"])
        wrong = cases[cases.after != cases.label]
        with st.expander(f"Curated cases still wrong ({len(wrong)} of {len(cases)})"):
            st.dataframe(
                wrong.rename(
                    columns={
                        "review_text": "Review",
                        "label": "Expected",
                        "before": "Before",
                        "after": "Now",
                    }
                ),
                hide_index=True,
            )
            st.caption(
                "These short phrases fail because a word-frequency model cannot read sentence "
                "structure, and in this data “hate” often appears in positive reviews. They "
                "were not used to tune anything."
            )


def model_info(predictor, report: dict | None) -> None:
    """Facts read from the artifact, its metadata and the hash-matched report only."""
    metadata = predictor.metadata
    vectorizer = predictor.model.named_steps.get("tfidf")
    ngrams = getattr(vectorizer, "ngram_range", None)
    with st.container(border=True):
        st.header("Model information")
        facts = {
            "Model": f"{metadata.get('selected_model', 'unknown')} classifier in one "
            "scikit-learn Pipeline (TF-IDF → classifier)",
            "Calibration": metadata.get("calibration", "not recorded"),
            "Labels": ", ".join(badge(label) for label in predictor.model.classes_),
            "Artifact": f"SHA-256 {metadata['artifact_sha256'][:12]}… · created "
            f"{str(metadata.get('created_utc', 'unknown'))[:10]}",
            "Features": "Word TF-IDF"
            + (f", n-grams {ngrams[0]}–{ngrams[1]}" if ngrams else "")
            + ". Inference runs locally; no review text leaves the server.",
            "Preprocessing": "HTML unescaped and tags removed, Unicode NFKC, lowercase, "
            "contractions expanded so negation survives (don't → do not), URLs and "
            "punctuation removed.",
            "Input rules": "1–20,000 characters with at least one two-letter word. Text with "
            "no familiar vocabulary returns unavailable, never a guessed label.",
            "Tentative threshold": f"Predictions below {unsure_threshold():.0%} predicted-class "
            "probability are flagged as tentative in the interface. Labels are unchanged.",
            "Library versions": ", ".join(
                f"{k} {v}" for k, v in metadata.get("versions", {}).items()
            ),
        }
        st.html(
            '<dl class="facts">'
            + "".join(
                f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in facts.items()
            )
            + "</dl>"
        )
        if not report:
            return
        audit = report.get("data_audit", {})
        sources = ", ".join(
            f"{s['filename']} ({s['rows']:,} rows)" for s in audit.get("sources", [])
        )
        st.markdown("**Training data provenance**")
        st.text(
            f"{sources}. After removing {audit.get('duplicate_rows_removed', 0):,} duplicates "
            f"and {audit.get('conflicting_rows_removed', 0):,} conflicting rows, "
            f"{audit.get('clean_rows', 0):,} reviews remained. Labels come from star ratings "
            "(low → negative, middle → neutral, high → positive)."
        )
        neutral = report["test"]["per_class"].get("neutral", {})
        st.markdown("**Known limitations**")
        st.text(
            f"Neutral reviews are hard for this model (training-run test recall "
            f"{neutral.get('recall', 0):.1%} before the decision rule; see the rule's "
            "evaluation for the current figure). Sarcasm, mixed opinions and non-English text "
            "are unreliable. Training data is mostly food reviews; the consumer-electronics "
            "slice is small. Rating-derived labels are themselves noisy."
        )


def evaluation(report: dict, directory: Path) -> None:
    metrics = report["test"]
    baseline = metrics.get("majority_baseline") or {}
    intervals = metrics.get("confidence_intervals") or {}
    with st.container(border=True):
        st.header("Original training-run evaluation")
        st.caption(
            f"The training run's own test split of {metrics['support']:,} reviews, scored once "
            "after model selection with the most-probable-class decision (before the decision "
            "rule above existed). Saved evaluation verified against the artifact hash."
        )
        a, b, c, d = st.columns(4)
        a.metric("Held-out accuracy", f"{metrics['accuracy']:.2%}", border=True)
        b.metric("Macro F1", f"{metrics['macro_f1']:.3f}", border=True)
        if baseline:
            c.metric(
                "Baseline accuracy",
                f"{baseline['accuracy']:.2%}",
                border=True,
                help="Always predicting the most common class. Accuracy alone flatters "
                "imbalanced data.",
            )
            d.metric("Baseline macro F1", f"{baseline['macro_f1']:.3f}", border=True)
        if intervals.get("accuracy"):
            st.caption(
                f"95% bootstrap intervals: accuracy [{intervals['accuracy']['lower']:.2%}, "
                f"{intervals['accuracy']['upper']:.2%}], macro F1 "
                f"[{intervals['macro_f1']['lower']:.3f}, {intervals['macro_f1']['upper']:.3f}] "
                f"({intervals.get('resamples', '?')} resamples; conditional on this fitted model)."
            )
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
            st.altair_chart(
                charts.confusion(metrics["confusion_matrix"], metrics["class_order"]),
                width="stretch",
            )
            st.dataframe(
                pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=metrics["class_order"],
                    columns=metrics["class_order"],
                )
            )
        with right:
            st.markdown("**Per-class results**")
            st.dataframe(
                pd.DataFrame(
                    {k: v for k, v in metrics["per_class"].items() if k in metrics["class_order"]}
                ).T
            )
            st.caption(
                "Neutral reviews are difficult for this model. Accuracy alone hides class imbalance."
            )
            sources = [
                {
                    "Source": name,
                    "Reviews": result["support"],
                    "Accuracy": result["accuracy"],
                    "Macro F1": result["macro_f1"],
                    "Low support": "yes" if result.get("low_support") else "no",
                }
                for name, result in report.get("per_source", {}).items()
            ]
            if sources:
                st.markdown("**By source** (within the same mixed test set)")
                st.dataframe(pd.DataFrame(sources), hide_index=True)
        figure = directory / "figures/calibration.png"
        if figure.exists():
            with st.expander("Reliability (calibration) curve"):
                st.image(str(figure))
                st.caption("How often predictions at each probability were right on the test set.")


def current_page() -> str:
    """Session state first, then a bookmarked ?page= link, then the analyzer."""
    if st.session_state.get("page") not in PAGES:
        wanted = st.query_params.get("page")
        st.session_state.page = next((p for p in PAGES if slug(p) == wanted), DEFAULT_PAGE)
    return st.session_state.page


def sidebar_nav(page: str) -> None:
    for group, pages in NAV.items():
        st.html(f'<div class="nav-group">{html.escape(group)}</div>')
        for name in pages:
            st.button(
                name,
                key=f"nav_{slug(name)}",
                icon=ICONS[name],
                type="primary" if name == page else "tertiary",
                width="stretch",
                on_click=go,
                args=(name,),
                help=PAGES[name],
            )


def chips(mode: str) -> str:
    meta = st.session_state.get("dataset_meta")
    frame = st.session_state.get("dataset")
    data = f"{meta['name']} · {len(frame):,} rows" if meta and frame is not None else "none loaded"
    return (
        f'<div class="chips"><span class="chip">Mode: {html.escape(mode)}</span>'
        f'<span class="chip">Dataset: {html.escape(data)}</span></div>'
    )


def render(predictor) -> None:
    keep_widget_state()
    page = current_page()
    with st.sidebar:
        st.html(
            '<div class="brand"><span class="brand-mark" aria-hidden="true">✦</span>'
            "<div><strong>Sentix AI</strong><span>Product review intelligence</span></div></div>"
        )
        sidebar_nav(page)
        st.divider()
        mode = st.radio(
            "Analysis mode",
            ["Fast Mode", "Deep Mode"],
            key="analysis_mode",
            on_change=clear_mode_results,
            help="Deep Mode defers only when the production predicted-class probability is below 0.80. There is no separate neutral override.",
        )
        if mode == "Deep Mode":
            st.caption(
                "Deep Mode sends reviews the Fast model is less than 80% sure of to an int8 "
                "transformer. In our test it found more neutral reviews but missed more "
                "negative ones, and was not better overall. The first transformer review "
                "loads the model (and downloads about 244 MB unless a verified local copy is "
                "configured). If Deep Mode is unavailable, results say so and use Fast Mode."
            )
        st.toggle(
            "Show Senti",
            key="show_senti",
            help="The animated companion on the Review Analyzer.",
            **initial("show_senti", value=True),
        )
        st.caption(
            "Deep Mode credits: [CardiffNLP Twitter-RoBERTa sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), CC BY 4.0. The int8 copy is a quantized, non-fine-tuned derivative."
        )
    selected = predictor
    progress_slot = st.empty()
    if mode == "Deep Mode":

        def progress(message, fraction):
            progress_slot.progress(fraction, text=message)

        try:
            selected = DeepPredictor(predictor, progress=progress)
        except Exception:
            st.warning("Deep Mode configuration is unavailable. Fast Mode is active.")
    group = next(g for g, pages in NAV.items() if page in pages)
    head, status = st.columns([3, 1.4], vertical_alignment="bottom")
    head.html(
        f'<div class="page-head"><span class="eyebrow">Sentix AI · {html.escape(group)}</span>'
        f"<h1>{html.escape(TITLES.get(page, page))}</h1><p>{html.escape(PAGES[page])}</p></div>"
    )
    # Filled after the page runs, since a page may replace the dataset.
    status_slot = status.empty()
    try:
        if page == "Overview":
            dashboard.overview(predictor)
        elif page == "Review Analyzer":
            live_review(selected)
        elif page == "Review Explorer":
            dashboard.explorer(predictor)
        elif page == "History":
            dashboard.history(predictor)
        elif page == "Model & Evaluation":
            st.caption("Saved production-model evaluation (Fast Mode).")
            model_lab(predictor)
        else:
            module = {"Batch Analysis": "bulk", "Explain": "explain", "Guess the Mood": "game"}[
                page
            ]
            if module == "explain" and mode == "Deep Mode":
                st.caption(
                    "This view explains the Fast Mode model only, not a deferred transformer result."
                )
            importlib.import_module(f"sentix.{module}").render(
                predictor if module == "explain" else selected
            )
    except Exception:
        st.info("Senti couldn't open this feature. Your live review is still available.")
    progress_slot.empty()
    status_slot.html(chips(mode))
