from flask import Flask, request, render_template
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import nltk

# Download necessary NLTK data files
nltk.download("stopwords")
nltk.download("punkt")

# Initialize Flask application
app = Flask(__name__)

# Load the sentiment analysis model and tokenizer
model = load_model("sentiment_analysis_model.h5")
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(
    pd.read_csv("sentiment_analysis.csv", encoding="latin1")["selected_text"].fillna("")
)

# Text preprocessing functions
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\d+", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Route to handle home page
@app.route("/")
def home():
    return render_template("index.html")


# Route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"].strip()  # Strip leading/trailing whitespace

    if text == "":
        text = "please provide input"
        sentiment = "Neutral"
    else:
        processed_text = preprocess_text(text)
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=100)
        prediction = model.predict(padded_sequence)
        sentiment = ["Negative", "Neutral", "Positive"][prediction.argmax(axis=1)[0]]

    return render_template("result.html", text=text, sentiment=sentiment)


if __name__ == "__main__":
    app.run(debug=True)
