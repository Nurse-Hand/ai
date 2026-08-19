from fastapi.testclient import TestClient

from main import app


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert "speakerCount" in payload
    assert "localSttModelDir" not in payload
    assert "dataDir" not in payload
    assert "tmpDir" not in payload


def test_speakers_list_empty_ok():
    client = TestClient(app)
    response = client.get("/api/speakers")
    assert response.status_code == 200
    assert "speakers" in response.json()


def test_legacy_http_exception_keeps_fastapi_detail_shape():
    client = TestClient(app)
    response = client.delete("/api/speakers/nonexistent")
    assert response.status_code == 404
    assert response.json() == {"detail": "speaker not found"}


def test_legacy_validation_keeps_fastapi_detail_shape():
    client = TestClient(app)
    response = client.post("/api/diarization/analyze")
    assert response.status_code == 422
    assert isinstance(response.json().get("detail"), list)
    assert "error" not in response.json()
