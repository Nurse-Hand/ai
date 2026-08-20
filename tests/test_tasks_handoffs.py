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


def test_extract_tasks_groups_same_patient_same_title(monkeypatch):
    client = _client(monkeypatch)
    request_id = str(uuid.uuid4())
    body = {
        "requestId": request_id,
        "evidence": [
            {
                "sourceId": "ev-1",
                "patientId": "p1",
                "summary": "산소포화도 확인이 필요합니다",
                "workDate": "2026-08-20",
            },
            {
                "sourceId": "ev-2",
                "patientId": "p1",
                "summary": "산소 라인 상태도 같이 봐야 합니다",
                "workDate": "2026-08-20",
            },
        ],
    }
    response = client.post(
        "/internal/v1/tasks/extract",
        headers={"X-Internal-Token": os.environ["INTERNAL_API_TOKEN"]},
        json=body,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["requestId"] == request_id
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["title"] == "산소 상태 확인"
    assert data["candidates"][0]["evidenceSourceIds"] == ["ev-1", "ev-2"]


def test_extract_tasks_allows_null_patient_id(monkeypatch):
    client = _client(monkeypatch)
    body = {
        "requestId": str(uuid.uuid4()),
        "evidence": [
            {
                "sourceId": "ev-1",
                "patientId": None,
                "summary": "창고 아세톤 개수 확인 필요",
                "workDate": "2026-08-20",
            }
        ],
    }
    response = client.post(
        "/internal/v1/tasks/extract",
        headers={"X-Internal-Token": os.environ["INTERNAL_API_TOKEN"]},
        json=body,
    )
    assert response.status_code == 201
    assert response.json()["candidates"][0]["patientId"] is None


def test_precheck_handles_no_candidates(monkeypatch):
    client = _client(monkeypatch)
    request_id = str(uuid.uuid4())
    body = {
        "requestId": request_id,
        "draftId": "draft-1",
        "patientId": "p1",
        "draftItems": [],
        "candidateEvidence": [],
        "openTasks": [],
    }
    response = client.post(
        "/internal/v1/handoffs/precheck",
        headers={"X-Internal-Token": os.environ["INTERNAL_API_TOKEN"]},
        json=body,
    )
    assert response.status_code == 201
    assert response.json() == {"requestId": request_id, "verificationItems": []}


def test_precheck_flags_missing_evidence_topic(monkeypatch):
    client = _client(monkeypatch)
    request_id = str(uuid.uuid4())
    body = {
        "requestId": request_id,
        "draftId": "draft-1",
        "patientId": "p1",
        "draftItems": [{"topic": "VITAL_SIGNS", "summary": "SpO2 안정적"}],
        "candidateEvidence": [
            {
                "evidenceId": "ev-1",
                "topic": "RESPIRATION",
                "handoffSection": "호흡",
                "structuredFacts": {"symptom": "기침"},
                "importanceFlags": ["follow_up_needed"],
                "requiresNurseConfirmation": False,
            }
        ],
        "openTasks": [],
    }
    response = client.post(
        "/internal/v1/handoffs/precheck",
        headers={"X-Internal-Token": os.environ["INTERNAL_API_TOKEN"]},
        json=body,
    )
    assert response.status_code == 201
    items = response.json()["verificationItems"]
    assert len(items) == 1
    assert items[0]["type"] == "MISSING_HANDOFF_ITEM"


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
