import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Resolve paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "next_word_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")

# Load pre-trained tokenizer
with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

# Load pre-trained model
model = load_model(MODEL_PATH)
max_seq_len = 5   # small value works fine

def predict_next_word(text_input):
    token_list = tokenizer.texts_to_sequences([text_input])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    predicted_word = np.argmax(predicted)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word:
            return word

while True:
    text = input("Enter sentence: ")
    print("Next word:", predict_next_word(text))
