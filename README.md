# 🤖 Next Word Predictor AI

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end Deep Learning application that predicts the next word in an input sequence. Built using a **Long Short-Term Memory (LSTM)** neural network architecture trained in Keras/TensorFlow, served through a high-performance **FastAPI** backend, and driven by a sleek, modern glassmorphic web portal.

This project is fully optimized for performance, modularity, and rapid cloud deployment, making it an excellent demonstration of production-level Deep Learning integration.

---

## 🌟 Features
- **Accurate Next Word Predictions**: Powered by an LSTM recurrent neural network designed for sequential text data.
- **Ultra-Fast Backend Gateway**: Powered by FastAPI, delivering predictions with sub-10ms server response times.
- **Modern Responsive Portal**: A visually stunning frontend featuring custom glassmorphism, responsive grid structures, loading feedback, and typing states.
- **Gateway Switcher**: Instantly switch the target API gateway between the live cloud server and localhost directly from the UI settings.
- **Performance Optimized Serialization**: Pre-fits and saves the NLP Tokenizer as a pickle (`.pkl`) asset, avoiding resource-heavy re-computation on API startup.
- **Built-in Inference Analytics**: Live network status and end-to-end inference latency tracker in milliseconds.

---

## 🛠️ Tech Stack

- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism, custom Outfit/Jakarta typography), Vanilla JavaScript (Asynchronous Fetch API).
- **Backend**: FastAPI (Python), Uvicorn (High-performance ASGI server), Pydantic (Strong request validation).
- **AI/ML**: TensorFlow / Keras (Neural weights model), NumPy (Vector decoding), Scikit-Learn/NLP Tokenizer (Word encoding & text preprocessing).
- **Deployment & Infra**: Render (Backend service), Vercel/Netlify (Frontend static pages).
- **Development Tools**: Python-dotenv (env management), Git (Version control).

---

## 📐 Architecture Overview

The system takes string sequences from the frontend client, tokenizes and pads the arrays to match the model input shape, feeds them into the trained LSTM model, decodes the maximum probability distribution indices, and returns the corresponding token string.

```mermaid
graph TD
    A[Frontend Client (Browser)] -->|POST /predict | B[FastAPI Gateway]
    B -->|Preprocess Input| C[Loaded Tokenizer (tokenizer.pkl)]
    C -->|Pad Sequence| D[LSTM Input Tensor (Shape: 1x4)]
    D -->|Inference| E[Keras LSTM Model (next_word_model.h5)]
    E -->|Probability Distribution| F[Argmax Decoding]
    F -->|Return Predicted Word| B
    B -->|JSON Response| A
```

### Recurrent Model Layers:
1. **Embedding Layer**: Transforms integer word tokens into 64-dimensional dense vectors.
2. **LSTM Layer**: Recurrent layer with 100 units, capturing sequential dependency patterns and semantic context.
3. **Dense Layer (Softmax)**: Computes a probability distribution across the entire vocabulary size.

---

## 📂 Project Structure

Below is the optimized, clean, and industry-standard folder hierarchy of the project:

```
next-word-ai/
├── .env.example               # Environment variables template
├── .gitignore                 # Exclusions for cache, venv, and binary checkpoints
├── README.md                  # Professional developer documentation
├── data/                      # Dataset corpus
│   └── corpus.txt             # Text dataset used for model training
├── models/                    # Model binary storage
│   ├── next_word_model.h5     # Serialized TensorFlow LSTM weights
│   └── tokenizer.pkl          # Serialized fitted NLP tokenizer
├── backend/                   # FastAPI Server Service
│   ├── main.py                # Main server script (endpoints, CORS, validation)
│   ├── requirements.txt       # Production dependencies
│   ├── runtime.txt            # Heroku/Render Python environment runtime version
│   └── start.sh               # Startup shell script for ASGI execution
├── frontend/                  # Web Client
│   └── index.html             # Revamped glassmorphism portal UI
└── ml/                        # Machine Learning Pipeline
    ├── train.py               # Dataset processing & LSTM model training script
    └── predict.py             # Standalone command-line inference utility
```

---

## 🖼️ Screenshots

> [!NOTE]
> Add visual screenshots of the interface here when deployed!

| Main Landing Interface (Glassmorphic Card) | Live Prediction & Latency Stats |
| --- | --- |
| ![Landing Page Placeholder](https://via.placeholder.com/600x350/0b0f19/00ffaa?text=Next+Word+AI+Landing+UI) | ![Stats Interface Placeholder](https://via.placeholder.com/600x350/0b0f19/8b5cf6?text=Prediction+Metrics+UI) |

---

## 🚀 Installation

Follow these steps to configure and run the application locally.

### Prerequisites
- **Python 3.10.x** installed.
- **Git** installed.

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server using Uvicorn:
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```
   *The Swagger interactive API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).*

### Frontend Setup
1. The frontend consists of a static file `frontend/index.html`.
2. You can open `frontend/index.html` directly in any web browser.
3. Switch the "API Gateway" dropdown to **Localhost (Port 8000)** to test predictions against your local FastAPI service.

### Environment Variables
Duplicate the environment template to configure variables:
```bash
cp .env.example .env
```
Inside `.env`:
```ini
HOST=127.0.0.1
PORT=8000
NEXT_PUBLIC_API_URL=https://next-word-ai.onrender.com
```

---

## 💡 Usage

1. Type a seed phrase in the text input box (e.g. `"deep learning is"` or `"i love coding"`).
2. Click the **Predict Next Word** button or press `Enter`.
3. The interface displays the predicted word (e.g. `"powerful"` or `"in"`) alongside network response latency and status.
4. Use the chip suggestions below the input box to instantly run test phrases.

---

## 🔌 API Endpoints

### 1. Health Probe
- **Method**: `GET`
- **Path**: `/`
- **Response**:
  ```json
  {
    "status": "healthy",
    "api": "Next Word Prediction AI API",
    "version": "1.0.0",
    "documentation": "/docs"
  }
  ```

### 2. Predict Next Word
- **Method**: `POST`
- **Path**: `/predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "text": "deep learning is"
  }
  ```
- **Response Body**:
  ```json
  {
    "input": "deep learning is",
    "next_word": "powerful"
  }
  ```

---

## 🧠 AI/ML Components

- **Model Type**: Sequential Recurrent Neural Network (RNN) using Long Short-Term Memory.
- **Tokenizer**: A fitted Keras `Tokenizer` mapping characters/words to integers based on frequency.
- **Inference Preprocessing**: Takes the incoming text, converts it to sequence integers via the deserialized `tokenizer.pkl`, and pads it using pre-padding to fit the fixed input sequence width (max sequence length - 1, which equals 4).
- **Prediction Postprocessing**: Performs a forward pass through `model.predict()`, fetches the index with the maximum probability via `np.argmax`, and scans the vocabulary dictionary map to extract the corresponding string token.

---

## 🛡️ Security Features
- **Pydantic Validation**: Strict length validation (`min_length=1`) on text fields to prevent empty or garbage payloads.
- **CORS Handling**: Configuration settings using FastAPI CORSMiddleware to control cross-origin requests securely.
- **Graceful Error Handling**: Input sanitizer checks prevent invalid, blank, or symbol-only requests from crashing the model thread.

---

## ⚡ Performance Optimizations
- **Tokenizer Pre-Serialization**: Solves the performance bottleneck of opening the corpus file and fitting the tokenizer on the fly during API start/requests. Loads a static serialized binary pickle (`tokenizer.pkl`) instantly.
- **Relative Path Resolution**: Resolves all references relative to the python script locations dynamically, ensuring the codebase is system-independent.
- **Low Memory Overhead**: Clean garbage collection and thread-safe loading of tensorflow weights.

---

## 🌐 Deployment Guide

### Deploying the Backend on Render
1. Create a new **Web Service** on Render.
2. Link your GitHub repository.
3. Configure the settings:
   - **Root Directory**: `backend` (or leave empty and run from root, configuring start script accordingly)
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `bash start.sh`
4. Set environment variable `PORT` in Render dashboard.

### Deploying the Frontend on Netlify/Vercel
1. Select the `frontend` folder as the publish directory.
2. Deploy as a static webpage. No build command is required.

---

## 🗺️ Future Improvements
- [ ] Expand the text corpus from simple programming statements to Wikipedia/Gutenberg datasets.
- [ ] Transition model from single LSTM to stacked bidirectional LSTM or a Transformer GPT-2 decoder.
- [ ] Build auto-suggest dropdowns directly inline as the user types (real-time typing predictions).

---

## 🧗 Challenges Faced

### Tokenizer Re-initialization Overhead
*Challenge*: The prototype loaded `data.txt` and fitted a new Tokenizer on *every single request* which degraded response speeds to >300ms.
*Solution*: Serialized the fitted Tokenizer object during training stage (`tokenizer.pkl`) and loaded it directly in the app. Latency plummeted to under 10ms.

---

## 🎓 Learning Outcomes
- Advanced serialization practices for multi-component ML applications.
- Serving deep learning pipelines as asynchronous REST APIs using FastAPI.
- Frontend optimization techniques (handling state transitions, measuring API speed metrics).

---

## 🤝 Contributing
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) details.

---

## ✍️ Author
**Subham**
- Professional Software Engineer & ML Enthusiast

---

## 📞 Contact
- **GitHub**: [github.com/subham1533](https://github.com/subham1533)
- **LinkedIn**: [LinkedIn Profile](https://linkedin.com)
- **Portfolio**: [Portfolio Site](https://portfolio.com)
