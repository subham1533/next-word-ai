import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Resolve paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "next_word_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")

# Load pre-trained tokenizer
try:
    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)
except FileNotFoundError:
    raise RuntimeError(f"Tokenizer not found at {TOKENIZER_PATH}. Run training first.")

# Load pre-trained model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {str(e)}")

max_seq_len = 5

app = FastAPI(
    title="Next Word Prediction AI API",
    description="A deep learning API powered by FastAPI and TensorFlow LSTM model to predict the next word in a sequence.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="Input sequence/sentence for prediction", json_schema_extra={"example": "deep learning is"})

def predict_next_word(text_input):
    token_list = tokenizer.texts_to_sequences([text_input])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

@app.get("/")
def home():
    return {
        "status": "healthy",
        "api": "Next Word Prediction AI API",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.post("/predict")
def predict(data: TextInput):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty or only spaces.")
    try:
        word = predict_next_word(data.text)
        return {
            "input": data.text,
            "next_word": word if word else "[Unrecognized]"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
