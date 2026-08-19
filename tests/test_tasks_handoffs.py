import os
import uuid

from fastapi.testclient import TestClient

from main import app


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv("INTERNAL_API_TOKEN", "test-token")
    return TestClient(app)


def test_internal_endpoints_reject_missing_auth(monkeypatch):
    client = _client(monkeypatch)
    response = client.post(
        "/internal/v1/tasks/prioritize",
        json={"requestId": str(uuid.uuid4()), "tasks": [], "patientRisk": [], "now": "2026-08-19T00:00:00Z"},
    )
    assert response.status_code == 401


def test_prioritize_tasks_echoes_request_id(monkeypatch):
    client = _client(monkeypatch)
    request_id = str(uuid.uuid4())
    body = {
        "requestId": request_id,
        "tasks": [{"taskId": "t1", "patientId": "p1", "title": "산소포화도 재측정", "carriedOver": True}],
        "patientRisk": [],
        "now": "2026-08-19T00:00:00Z",
    }
    response = client.post(
        "/internal/v1/tasks/prioritize",
        headers={"X-Internal-Token": os.environ["INTERNAL_API_TOKEN"]},
        json=body,
    )
    assert response.status_code == 201
    assert response.json()["requestId"] == request_id


def test_precheck_handles_empty_batch(monkeypatch):
    client = _client(monkeypatch)
    request_id = str(uuid.uuid4())
    body = {"requestId": request_id, "patients": [], "lookbackShifts": 3}
    response = client.post(
        "/internal/v1/handoffs/precheck",
        headers={"X-Internal-Token": os.environ["INTERNAL_API_TOKEN"]},
        json=body,
    )
    assert response.status_code == 201
    assert response.json() == {"requestId": request_id, "items": []}


def test_generate_handles_no_evidence(monkeypatch):
    client = _client(monkeypatch)
    request_id = str(uuid.uuid4())
    body = {"requestId": request_id, "patientId": "p1", "roundingSessionId": "round-1", "evidences": []}
    response = client.post(
        "/internal/v1/handoffs/generate",
        headers={"X-Internal-Token": os.environ["INTERNAL_API_TOKEN"]},
        json=body,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["patientId"] == "p1"
    assert data["roundingSessionId"] == "round-1"
    assert data["items"] == []
