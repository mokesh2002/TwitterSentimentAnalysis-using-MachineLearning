import pickle
import re  # Add this import
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

app = Flask(__name__)

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vector.pkl', 'rb') as vector_file:
    vectorizer = pickle.load(vector_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']

        # Preprocess the text
        text = text.lower()
        text = re.sub(r'https\S+|www\S+https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)

        # Vectorize the text
        text_vectorized = vectorizer.transform([text])

        # Predict sentiment
        sentiment = model.predict(text_vectorized)

        return render_template('index.html', text=text, sentiment=sentiment[0])

if __name__ == '__main__':
    app.run(debug=True)
