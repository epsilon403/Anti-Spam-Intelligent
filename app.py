import os
import re
import joblib
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

VECTORIZER_PATH = "tfidf_vectorizer.joblib"
MODEL_PATH = "svm_model.joblib"


def ensure_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


def preprocess_text(text, lemmatizer, stop_words):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', 'url', text)
    text = re.sub(r'\d+', 'number', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    normalized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.strip()]
    return " ".join(normalized_tokens)


def main():
    st.title("Anti-Spam Tester")
    st.write("Load the saved TF-IDF vectorizer and SVM model, then test individual emails.")

    ensure_nltk()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH)):
        st.error("Model or vectorizer not found. Run 'export_model.py' first to create them.")
        st.stop()

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    user_input = st.text_area("Paste email text (subject+body) to classify:")

    if st.button("Predict"):
        processed = preprocess_text(user_input, lemmatizer, stop_words)
        X = vectorizer.transform([processed])
        pred = model.predict(X)[0]
        prob = None
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[0]

        label = 'spam' if int(pred) == 1 else 'ham'
        st.write(f"**Prediction:** {label}")
        if prob is not None:
            st.write(f"**Probabilities:** ham={prob[0]:.4f}, spam={prob[1]:.4f}")


if __name__ == '__main__':
    main()
