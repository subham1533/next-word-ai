from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load text & tokenizer
with open("data.txt", "r") as file:
    text = file.read().lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

max_seq_len = 5
model = load_model("next_word_model.h5")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str

def predict_next_word(text_input):
    token_list = tokenizer.texts_to_sequences([text_input])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

@app.get("/")
def home():
    return {"message": "Next Word Prediction API running"}

@app.post("/predict")
def predict(data: TextInput):
    word = predict_next_word(data.text)
    return {"next_word": word}
