import pickle
import nltk
import streamlit as st
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Open the .pkl file in read binary mode ('rb')
with open('sentiment.pkl', 'rb') as f:
    # Load the data from the file
    model = pickle.load(f)

# Load Tokenizer
tokenizer = Tokenizer()

# Maximum length of sequences
max_len = 100

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Streamlit app
st.title("Sentiment Analysis App")

# User input in the dedicated textarea
user_input = st.text_area("Enter your text:")

# Tokenize and pad the user input
if user_input:
    user_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_sequence, maxlen=max_len)

    # Predict sentiment using the trained model
    if st.button("Analyze Sentiment"):
        prediction = model.predict(user_input_padded)

        # Analyze sentiment using VADER
        vader_score = sia.polarity_scores(user_input)

        # Display results in the Streamlit app
        st.write(f"Neural Network Prediction: {'Positive' if prediction >= 0.5 else 'Negative'}")
        st.write(f"VADER Sentiment Analysis: {'Positive' if vader_score['compound'] >= 0 else 'Negative'}")
else:
    st.warning("Please enter your text.")
