import streamlit as st
import pickle

# Set the page configuration
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

# Load the models and vectorizer
@st.cache_resource
def load_vectorizer():
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model():
    with open("models/svm_model.pkl", "rb") as f:
        return pickle.load(f)

vectorizer = load_vectorizer()
model = load_model()

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .navbar {
        background-color: #f4f4f4;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .navbar h3 {
        margin: 0;
        color: #333;
        font-weight: bold;
        text-align: center;
    }
    .sentiment-box {
        text-align: center;
        margin-top: 30px;
    }
    .positive {
        color: #28a745;
    }
    .negative {
        color: #dc3545;
    }
    .neutral {
        color: #6c757d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation bar-like header
st.markdown(
    """
    <div class="navbar">
        <h3>Sentiment Analyzer</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# App Title
st.markdown('<div class="title">Sentiment Analysis Tool</div>', unsafe_allow_html=True)

# Introduction text
st.write(
    "Welcome to the **Sentiment Analyzer**! This tool leverages a **Support Vector Machine (SVM)** model trained with **TF-IDF vectorization** to analyze the sentiment of your input text. The model can classify sentiments as **Positive**, **Negative**, or **Neutral**."
)

# Input Section
st.subheader("Enter Text for Sentiment Analysis")
user_input = st.text_area("Type or paste your text below:")


# Process the text and predict sentiment
if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Preprocess and make prediction
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)

        # Determine sentiment
        if prediction[0] == 1:
            sentiment = "Positive"
            sentiment_class = "positive"
            emoji = "😊"
        elif prediction[0] == 2:
            sentiment = "Negative"
            sentiment_class = "negative"
            emoji = "😞"
        else:
            sentiment = "Neutral"
            sentiment_class = "neutral"
            emoji = "😐"

        # Display sentiment
        st.markdown(
            f"""
            <div class="sentiment-box">
                <h2 class="{sentiment_class}">
                    Sentiment: {sentiment} {emoji}
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("Please enter some text to analyze.")

# Footer with project details
st.markdown(
    """
    <hr style="border:1px solid #ddd; margin-top: 30px;">
    <div style="text-align: center; font-size: 14px; color: #666;">
        © 2024 Sentiment Analyzer Project. Built with ❤️ using Python, Streamlit, and Machine Learning.
    </div>
    """,
    unsafe_allow_html=True,
)
