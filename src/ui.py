"""Evidence-driven Streamlit views."""
import pandas as pd
import streamlit as st
from src.predict import explain_review, export_csv, predict_reviews, read_batch, top_terms
from src.preprocessing import MAX_REVIEW_CHARS


def single_review(model):
    st.subheader("Analyze a review")
    examples = {"Write my own": "", "Positive example": "Excellent quality and easy to use. I would buy it again.",
                "Mixed example": "It works, but the quality is average for the price.",
                "Negative example": "It stopped working after two days. I am disappointed."}
    choice = st.selectbox("Try an example", list(examples))
    text = st.text_area("Review", value=examples[choice], height=150, max_chars=MAX_REVIEW_CHARS, key=f"review_{choice}")
    if st.button("Analyze review", type="primary"):
        try:
            with st.spinner("Analyzing review..."):
                result = predict_reviews(model, [text]).iloc[0]
            if result.status != "ok":
                st.warning("No terms matched the model vocabulary. Try a more detailed English review.")
                return
            st.success(f"Predicted sentiment: {result.sentiment.title()}")
            prefix = "probability_" if "model_probability" in result else "margin_"
            scores = {label: result[f"{prefix}{label}"] for label in model.classes_}
            if prefix == "probability_":
                st.metric("Model probability", f"{result.model_probability:.1%}")
                st.caption("Uncalibrated model probabilities are not a guarantee of correctness or a product rating.")
            else:
                st.caption("Decision margins compare class scores; they are not probabilities or confidence percentages.")
            st.bar_chart(pd.Series(scores, name="Score"))
            explanation = explain_review(model, text, result.sentiment)
            if not explanation.empty:
                st.markdown("**Words contributing to this class score**")
                st.bar_chart(explanation.set_index("term"))
                st.caption("TF-IDF × class coefficient. Positive values raise this class score; negative values lower it. The intercept is omitted. This explains a model score, not causality.")
        except ValueError as error:
            st.error(str(error))


def batch_reviews(model):
    st.subheader("Batch CSV prediction")
    st.write("Upload a UTF-8 CSV with a `review_text` column. Maximum: 1,000 reviews, 5 MB, 20,000 characters per review. Extra columns are ignored.")
    st.caption("Reviews are processed in server memory and are not deliberately saved by this app. Avoid uploading sensitive information to a public deployment.")
    st.download_button("Download example CSV", b'review_text\n"Great quality and fast delivery."\n"The item broke after one use."\n"It is average for the price."\n', "example_reviews.csv", "text/csv")
    uploaded = st.file_uploader("Review CSV", type=["csv"])
    if uploaded is not None:
        try:
            frame = read_batch(uploaded.getvalue())
            with st.spinner("Predicting batch..."):
                result = predict_reviews(model, frame.review_text)
            st.dataframe(result, hide_index=True)
            st.bar_chart(result.sentiment.value_counts().rename("Reviews"))
            st.caption("Distribution describes predictions for this upload, not measured customer satisfaction. No-known-term rows are marked unavailable.")
            st.download_button("Download predictions", export_csv(result), "sentiment_predictions.csv", "text/csv")
        except ValueError as error:
            st.error(str(error))


def performance(metadata):
    st.subheader("Model comparison and holdout evaluation")
    st.write("Selection uses mean macro F1 from training-only cross-validation. Each class receives equal weight, making neutral and negative performance visible.")
    comparison = pd.DataFrame(metadata["comparison"])
    st.bar_chart(comparison.set_index("model")[["cv_macro_f1", "cv_weighted_f1", "cv_accuracy"]])
    st.dataframe(comparison.drop(columns="best_parameters"), hide_index=True)
    st.caption("CV scores include model tuning; they are selection estimates. Standard deviation shows fold variability, not a confidence interval.")
    holdout = metadata["holdout"]
    for column, metric in zip(st.columns(3), ["accuracy", "macro_f1", "weighted_f1"]):
        column.metric(metric.replace("_", " ").title(), f"{holdout[metric]:.3f}")
    st.write(f"Selected model: **{metadata['selected_model']}**. Holdout: {metadata['test_rows']:,} reviews, evaluated after selection.")
    labels = holdout["class_order"]
    st.markdown("**Confusion matrix — rows: actual; columns: predicted**")
    matrix = pd.DataFrame(holdout["confusion_matrix"], index=labels, columns=labels)
    st.dataframe(matrix)
    report = pd.DataFrame({key: value for key, value in holdout["classification_report"].items() if isinstance(value, dict)}).T
    st.dataframe(report)


def data_view(metadata):
    st.subheader("Data quality and distribution")
    audit = metadata["data_audit"]
    st.write(f"{audit['input_rows']:,} input rows → {audit['clean_rows']:,} usable unique reviews → {metadata['sample_rows']:,} sampled reviews.")
    st.bar_chart(pd.Series(audit["class_distribution"], name="Cleaned full dataset"))
    st.dataframe(pd.DataFrame(metadata["split_class_distribution"]))
    st.markdown("**Data quality exclusions**")
    st.json({key: value for key, value in audit.items() if key.endswith("removed")})
    st.markdown("**Training review length quantiles (characters)**")
    st.bar_chart(pd.Series(metadata["training_length_quantiles"], name="Characters"))
    st.caption("Normalized exact duplicates are removed before splitting; all conflicting-label duplicate groups are excluded. Near-duplicate and product/author overlap are not eliminated.")
    st.dataframe(pd.DataFrame(audit["sources"]), hide_index=True)


def render(model, metadata):
    page = st.sidebar.radio("Explore", ["Overview", "Single review", "Batch CSV", "Data exploration", "Model performance", "Explainability", "About and model info"])
    st.sidebar.caption(f"Model: {metadata['selected_model']}\n\nEnglish product reviews • Three classes")
    if page == "Overview":
        st.subheader("Turn review text into a clearer feedback picture")
        st.write("Classify positive, negative, and neutral product reviews to support feedback triage. Explore the evidence behind the model before relying on its output.")
        for column, label, value in zip(st.columns(3), ["Training reviews", "Holdout macro F1", "Sentiment classes"], [f"{metadata['train_rows']:,}", f"{metadata['holdout']['macro_f1']:.3f}", "3"]):
            column.metric(label, value)
        st.markdown("**Start with a single review**, upload a CSV, or inspect model performance using the sidebar.")
        st.info("Labels come from star ratings: 1–2 negative, 3 neutral, 4–5 positive. A three-star review may be mixed rather than emotionally neutral.")
        st.write("The original university notebook and report remain in the repository. The app uses a separately tested, reproducible training pipeline.")
    elif page == "Single review":
        single_review(model)
    elif page == "Batch CSV":
        batch_reviews(model)
    elif page == "Data exploration":
        data_view(metadata)
    elif page == "Model performance":
        performance(metadata)
    elif page == "Explainability":
        st.subheader("Terms associated with each sentiment")
        label = st.selectbox("Sentiment class", list(model.classes_))
        terms = top_terms(model, label)
        if terms.empty:
            st.info("Coefficient explanations are available for linear models only.")
        else:
            st.bar_chart(terms.set_index("term"))
            st.caption("Largest positive class coefficients. Associations can reflect dataset biases and are not causal explanations. Single-review analysis shows local contributions.")
    else:
        st.subheader("About this model")
        st.write("Python • scikit-learn • TF-IDF word unigrams/bigrams • Streamlit")
        st.write("The model is trained offline; the app caches its complete preprocessing and classification pipeline. It is a portfolio application, not a monitored production service.")
        for limitation in metadata["limitations"]:
            st.write(f"• {limitation}")
        st.json({key: metadata[key] for key in ["selected_model", "created_utc", "python", "versions", "config", "artifact_sha256"]})
