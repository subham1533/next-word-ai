import sys
import os
from fastapi.testclient import TestClient

# Add project root to python path so backend module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert json_data["api"] == "Next Word Prediction AI API"

def test_predict_success():
    response = client.post("/predict", json={"text": "deep learning is"})
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["input"] == "deep learning is"
    assert "next_word" in json_data

def test_predict_empty():
    response = client.post("/predict", json={"text": ""})
    # Pydantic validates min_length=1, returning 422
    assert response.status_code == 422

def test_predict_whitespace():
    response = client.post("/predict", json={"text": "   "})
    # Handled inside the endpoint, returns 400
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()
