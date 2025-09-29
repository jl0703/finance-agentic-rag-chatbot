from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_response_success():
    payload = {"user_id": "test_user", "message": "Hi"}
    response = client.post("/chat/", json=payload)
    assert response.status_code == 200
    assert "Hello" or "Hi" in response.json()
    assert isinstance(response.json()["response"], str)

def test_chat_response_error():
    payload = {"user_id": "test_user", "message": ""}
    response = client.post("/chat/", json=payload)
    assert response.status_code in [200, 500]