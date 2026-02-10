import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# load tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

# reload tokenizer from training
with open("data.txt","r") as file:
    text = file.read().lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

model = load_model("next_word_model.h5")
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
