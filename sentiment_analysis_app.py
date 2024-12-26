from flask import Flask, request, render_template
import numpy as np
import joblib
import re

# Initialize Flask app
app = Flask(__name__)

# Function to preprocess the text
def preprocess_text(text):
    text = text.lower()  # Converting to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)  # Removing punctuation and numbers
    text = re.sub(r'\s+', ' ', text)  # Removing multiple spaces
    return text

# Load your trained model and TF-IDF vectorizer
model = joblib.load('C:\\Users\\sa304\\OneDrive\\Desktop\\Doctor-Review-Sentiment-Analysis\\Doctor-Review-Sentiment-Analysis\\logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('C:\\Users\\sa304\\OneDrive\\Desktop\\Doctor-Review-Sentiment-Analysis\\Doctor-Review-Sentiment-Analysis\\tfidf_vectorizer.pkl')

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        user_input = request.form['review_text']
        # Preprocess the input
        processed_text = preprocess_text(user_input)
        # Vectorize the input
        vectorized_text = tfidf_vectorizer.transform([processed_text])
        # Predict sentiment
        prediction = model.predict(vectorized_text)
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        return render_template('index.html', prediction_text=f'Predicted Sentiment: {sentiment}')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
z