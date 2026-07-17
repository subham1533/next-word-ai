# 🤖 Next Word Prediction AI Portal

A premium, full-stack deep learning application that uses a Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) to predict the next word in a sentence sequence in real-time. The project features a FastAPI back-end API for model inference, a Python-based machine learning training pipeline, and a modern, high-performance web interface built with glassmorphism styling in Vanilla CSS/HTML5.

---

## 📂 Project Architecture & Directory Structure

```
next-word-ai/
├── backend/
│   ├── main.py            # FastAPI prediction server
│   ├── requirements.txt   # Backend python dependencies
│   ├── runtime.txt        # Backend runtime specification
│   └── start.sh           # Script to launch the API server
├── data/
│   └── corpus.txt         # Text dataset used for model training
├── frontend/
│   └── index.html         # Responsive, glassmorphism UI portal
├── ml/
│   ├── train.py           # ML script to tokenize corpus & train LSTM
│   └── predict.py         # CLI-based real-time prediction loop
├── models/
│   ├── next_word_model.h5 # Trained Keras LSTM model
│   └── tokenizer.pkl      # Tokenizer vocabulary file
├── tests/
│   └── test_backend.py    # Pytest suite for FastAPI handlers
└── README.md              # Detailed documentation
```

---

## 🛠️ Technology Stack

1. **Front-end**:
   - **HTML5 & Vanilla JS (ES6+)**: High speed, zero dependencies, dynamic asynchronous API fetching.
   - **Vanilla CSS3**: Sleek styling featuring dark mode variables, glassmorphism (`backdrop-filter`), flexbox layouts, and micro-animations.
2. **Back-end**:
   - **FastAPI**: Extremely fast, asynchronous Python web framework powered by Starlette and Pydantic. Automatically generates interactive API docs (Swagger UI).
   - **Uvicorn**: High-performance ASGI server for hosting Python applications.
3. **Machine Learning & Deep Learning**:
   - **TensorFlow / Keras**: Deep learning library used to build, compile, and train the LSTM network.
   - **NumPy**: Multidimensional array processing for sequence handling.
   - **Pickle**: Serializes the fitted Tokenizer vocabulary.

---

## 🧠 Neural Network & Model Design

The model is built as a sequential deep learning architecture in Keras:
- **Embedding Layer**: Projects the sparse input sequence tokens into a dense vector space (dimension `64`), helping capture semantic similarities between words.
- **LSTM Layer (100 units)**: A recurrent layer containing memory gates to learn short and long-term sequential patterns and context in sentences.
- **Dense Output Layer**: Applies a `softmax` activation over the entire vocabulary to output a probability distribution for the next word.
- **Optimization**: Uses `categorical_crossentropy` loss and the `Adam` optimizer to train over 100 epochs.

---

## 🚀 How to Run the Project

### 1. Requirements & Setup
Make sure you have Python 3.10+ installed.

Install the required Python libraries:
```bash
pip install -r backend/requirements.txt
```

### 2. Train the Model
If you want to re-train the model or update it with new vocabulary inside `data/corpus.txt`, run the training script:
```bash
python ml/train.py
```
This updates `models/next_word_model.h5` and `models/tokenizer.pkl`.

### 3. Run the Backend API
Start the FastAPI server:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
- Interactive Docs: View Swagger documentation at `http://localhost:8000/docs`.

### 4. Run Frontend Portal
Double-click `frontend/index.html` to open it in any web browser, or serve it using a local dev server.
- The web portal allows you to select local API endpoints (`http://localhost:8000`) or cloud deployed gateways (`https://next-word-ai.onrender.com`) to run predictions instantly.

---

## 🧪 Testing

We use `pytest` for automated backend routing and inference verification:
```bash
pytest
```
Tests assert the correctness of API status, normal predictions, input validations, and error responses.
