import numpy as np
import nltk
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Resolve paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "corpus.txt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "next_word_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")

# read dataset
with open(DATA_PATH, "r", encoding="utf-8") as file:
    text = file.read().lower()

# tokenize words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

# save tokenizer
os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
with open(TOKENIZER_PATH, "wb") as file:
    pickle.dump(tokenizer, file)
print(f"Tokenizer saved to {TOKENIZER_PATH}")

word_index = tokenizer.word_index
total_words = len(word_index) + 1

print("Total words:", total_words)




input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

print("Total sequences:", len(input_sequences))




max_seq_len = max([len(seq) for seq in input_sequences])

input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]


from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=total_words)





from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()

model.add(Embedding(total_words, 64, input_length=max_seq_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


model.fit(X, y, epochs=100, verbose=1)

model.save(MODEL_PATH)
print(f"Model trained and saved to {MODEL_PATH}!")
