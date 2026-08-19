from fastapi.testclient import TestClient

from main import app


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert "speakerCount" in payload


def test_speakers_list_empty_ok():
    client = TestClient(app)
    response = client.get("/api/speakers")
    assert response.status_code == 200
    assert "speakers" in response.json()
