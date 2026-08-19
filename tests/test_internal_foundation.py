import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.errors import InferenceFailure, InferenceFailureCode
from app.llm import call_structured
from app.schemas import PrioritizeTasksResponse
from main import app


def test_internal_auth_uses_internal_token(monkeypatch):
    monkeypatch.setenv("INTERNAL_TOKEN", "foundation-token")
    get_settings.cache_clear()
    response = TestClient(app).post(
        "/internal/v1/tasks/prioritize",
        json={"requestId": "request", "tasks": [], "patientRisk": [], "now": "now"},
    )
    assert response.status_code == 401
    get_settings.cache_clear()


def test_missing_openai_key_never_returns_legacy_stub(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    stub = PrioritizeTasksResponse(requestId="request", results=[])
    with pytest.raises(InferenceFailure) as captured:
        call_structured("prompt", "payload", PrioritizeTasksResponse, stub)
    assert captured.value.code == InferenceFailureCode.UNAVAILABLE
    get_settings.cache_clear()
