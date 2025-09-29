from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_openai():
    response = client.get("/health/openai")
    assert response.status_code in [200, 503]
    assert "status" in response.json() or "detail" in response.json()

def test_health_redis():
    response = client.get("/health/redis")
    assert response.status_code == 200
    assert response.json()["status"] == "Healthy"

def test_health_qdrant():
    response = client.get("/health/qdrant")
    assert response.status_code == 200
    assert response.json()["status"] == "Healthy"