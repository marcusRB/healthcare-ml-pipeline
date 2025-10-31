import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)  # <-- positionally

@pytest.fixture(scope="module", autouse=True)
def setup_app():
    """Ensures FastAPI app and model startup event run before tests."""
    with TestClient(app) as c:
        yield c

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    print("✅ Root Response:", data)

def test_model_health():
    response = client.get("/model/health")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data
    assert data["model_loaded"] is True
    print("✅ Model Health:", data)

def test_predict_endpoint():
    response = client.post("/predict", params={"records": 10})
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "sample_predictions" in data
    assert len(data["sample_predictions"]) > 0
    print("✅ Predict Response Summary:", data["summary"])
