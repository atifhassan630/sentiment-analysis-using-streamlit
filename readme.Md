# Sentiment Analyzer using Streamlit

This project is a simple web application for sentiment analysis, built with Python, Streamlit, and machine learning. It uses a Support Vector Machine (SVM) model trained with TF-IDF vectorization to classify text as **Positive**, **Negative**, or **Neutral**.

## Features

- User-friendly web interface powered by Streamlit
- Predicts sentiment of user-input text
- Displays results with clear labels and emojis
- Custom styling for a modern look

## Getting Started

### Prerequisites

- Python 3.8 or higher
- All required packages listed in [requirements.txt](requirements.txt)

### Installation

1. Clone this repository:
   

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure the following files are present:
    - `models/svm_model.pkl`
    - `models/tfidf_vectorizer.pkl`

### Running the App

Start the Streamlit app with:
```sh
streamlit run app1.py
```

The app will open in your browser. Enter text and click "Analyze Sentiment" to see the result.

## Project Structure

```
.
├── app1.py
├── requirements.txt
├── sentiments.txt
├── models/
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
└── ...
```

- `app1.py`: Main Streamlit application
- `models/`: Contains the trained SVM model and TF-IDF vectorizer
- `sentiments.txt`: Example sentences for each sentiment

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

Built with ❤️ using Python, Streamlit, and Machine Learning.
