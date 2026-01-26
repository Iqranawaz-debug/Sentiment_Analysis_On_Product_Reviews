import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="Sentiment Analysis App",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------ CUSTOM STYLING ------------------ #
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextArea label {
        color: #212529 !important;
    }
    textarea {
        background-color: #ffffff !important;
        color: #212529 !important;
    }
    .custom-button {
        background-color: #007bff !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .custom-box {
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .positive-box {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .negative-box {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .neutral-box {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODELS ------------------ #
model = joblib.load(r"C:\Users\User\Documents\6th Semester\ML\project\Sentiment_app\voting_model.pkl")
vectorizer = joblib.load(r"C:\Users\User\Documents\6th Semester\ML\project\Sentiment_app\vectorizer.pkl")

# ------------------ TITLE ------------------ #
st.title("🧠 Sentiment Analysis App")
st.markdown("Analyze the sentiment of customer reviews using an ensemble ML model.")

# ------------------ INPUT ------------------ #
text_input = st.text_area("✍️ Enter a review below:", height=100)

def get_star_rating(confidence):
    stars = int(round(confidence / 20))  # 0–100 → 0–5 stars
    return "⭐" * stars + "☆" * (5 - stars)

if st.button("🔍 Analyze", type="primary"):
    if text_input.strip():
        vectorized_input = vectorizer.transform([text_input])
        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]
        confidence = np.max(probabilities) * 100
        emoji_icon = {"positive": "😊", "negative": "😞", "neutral": "😐"}[prediction]

        # ---- Show Sentiment Result ---- #
        if prediction == "positive":
            st.markdown(f"<div class='custom-box positive-box'>✅ Positive Review {emoji_icon}</div>", unsafe_allow_html=True)
        elif prediction == "negative":
            st.markdown(f"<div class='custom-box negative-box'>❌ Negative Review {emoji_icon}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='custom-box neutral-box'>ℹ️ Neutral Review {emoji_icon}</div>", unsafe_allow_html=True)

        # ---- Show Confidence & Stars ---- #
        st.markdown(f"**Model Prediction:** `{prediction}`")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.markdown(f"**Star Rating:** {get_star_rating(confidence)}")
        st.progress(confidence / 100)

        # ---- Pie Chart for Sentiment Scores ---- #
        st.subheader("📈 Sentiment Distribution")
        labels = ["Positive", "Negative", "Neutral"]
        colors = ["#28a745", "#dc3545", "#6c757d"]
        fig, ax = plt.subplots()
        ax.pie(probabilities, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        ax.axis("equal")
        st.pyplot(fig)
