import pickle
import nltk
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask,request,jsonify,render_template
app =Flask(__name__)

# Download NLTK data
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# Load stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# HTML and punctuation removal
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # remove HTML
    text = re.sub(f"[{string.punctuation}]", "", text)    # remove punctuation
    return text.lower()

# Tokenize, remove stopwords, lemmatize
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Load vectorizer and model
with open("vectorizer.pkl", 'rb') as f:
    vectorizer = pickle.load(f)

with open("model.pkl", 'rb') as f:
    model = pickle.load(f)
def model_prediction(text):
    text = [text]
    review = {1: "positive", 0: "negative"}

    cleaned = [clean_text(t) for t in text]
    preprocessed = [preprocess(t) for t in cleaned]
    X = vectorizer.transform(preprocessed)
    prediction = model.predict(X)
    return review[prediction[0]]

    

# # Sample input
# sample = ["movie is good"]

# # Step 1: Clean
# cleaned = [clean_text(text) for text in sample]

# # Step 2: Preprocess
# preprocessed = [preprocess(text) for text in cleaned]

# # Step 3: Vectorize
# X = vectorizer.transform(preprocessed)

# # Step 4: Predict
# prediction = model.predict(X)
# probability = model.predict_proba(X)
@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        print("Received POST request")  # Debug
        text_input = request.form['user_text']
        prediction_result = model_prediction(text_input)
        result = f"Sentiment: {prediction_result}"
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)




