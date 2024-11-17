
import os
import streamlit as st
import nltk
import pickle
import json
import numpy as np
import random
from tensorflow.keras.models import load_model
import google.generativeai as genai
from nltk.stem import WordNetLemmatizer

# Suppress TensorFlow and Absl warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load files
try:
    model_path = "/content/model_name (1).h5"  # Adjust to your file paths
    words_path = "/content/words(01) (1).pkl"
    classes_path = "/content/classes(01) (1).pkl"
    intents_path = "/content/datasetchatbot.json"  # You must upload this

    model = load_model(model_path)
    words = pickle.load(open(words_path, 'rb'))
    classes = pickle.load(open(classes_path, 'rb'))

    with open(intents_path, encoding="utf8") as f:
        intents = json.load(f)
except Exception as e:
    st.error(f"Error loading files: {str(e)}")

# Helper functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list):
    tag = intents_list[0]['intent']
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

def handle_request(user_message):
    prediction = predict_class(user_message)
    response = get_response(prediction)
    return response

# Streamlit UI
def chatbot_ui():
    st.title("Chatbot Application")
    st.write("Ask me anything!")

    user_input = st.text_input("Enter your message:")

    if st.button("Submit"):
        if user_input:
            response = handle_request(user_input)
            st.write("Chatbot Response:")
            st.write(response)
        else:
            st.write("Please enter a message to get started.")

# Run the Streamlit app
if __name__ == "__main__":
    chatbot_ui()
