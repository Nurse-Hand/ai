from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from openai import ContentFilterFinishReasonError, LengthFinishReasonError

import app.llm as llm_module
from app.auth import verify_internal_token
from app.config import get_settings
from app.errors import InferenceFailure, InferenceFailureCode
from app.llm import call_structured
from app.schemas import PrioritizeTasksResponse
from main import app, http_error_handler, inference_error_handler

foundation_app = FastAPI()
foundation_app.add_exception_handler(InferenceFailure, inference_error_handler)
foundation_app.add_exception_handler(HTTPException, http_error_handler)


@foundation_app.get("/internal/v1/__foundation__/auth", dependencies=[Depends(verify_internal_token)])
def foundation_auth_probe():
    return {"ok": True}


@foundation_app.get("/internal/v1/__foundation__/failure", dependencies=[Depends(verify_internal_token)])
def foundation_failure_probe(code: InferenceFailureCode):
    raise InferenceFailure(code)


@foundation_app.get("/internal/v1/__foundation__/llm", dependencies=[Depends(verify_internal_token)])
def foundation_llm_probe():
    return call_structured("system", "SENSITIVE_SOURCE_PAYLOAD", PrioritizeTasksResponse)


@pytest.fixture
def foundation_client(monkeypatch):
    monkeypatch.setenv("INTERNAL_TOKEN", "foundation-token")
    get_settings.cache_clear()
    with TestClient(foundation_app) as client:
        yield client
    get_settings.cache_clear()


def test_internal_token_unset_fails_closed_with_503(monkeypatch):
    monkeypatch.delenv("INTERNAL_TOKEN", raising=False)
    get_settings.cache_clear()
    response = TestClient(foundation_app).get("/internal/v1/__foundation__/auth")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "INTERNAL_AUTH_UNAVAILABLE"
    get_settings.cache_clear()


def test_wrong_internal_token_returns_401(foundation_client):
    response = foundation_client.get(
        "/internal/v1/__foundation__/auth", headers={"X-Internal-Token": "wrong-token"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_INTERNAL_TOKEN"


@pytest.mark.parametrize(
    ("code", "status"),
    [
        (InferenceFailureCode.TIMEOUT, 504),
        (InferenceFailureCode.RATE_LIMITED, 429),
        (InferenceFailureCode.UNAVAILABLE, 503),
        (InferenceFailureCode.INVALID_RESPONSE, 502),
    ],
)
def test_inference_failures_have_stable_status_and_safe_body(
    foundation_client, code, status
):
    response = foundation_client.get(
        "/internal/v1/__foundation__/failure",
        headers={"X-Internal-Token": "foundation-token"},
        params={"code": code.value},
    )
    assert response.status_code == status
    assert response.json() == {
        "error": {"code": code.value, "message": "AI inference failed."}
    }
    assert "foundation-token" not in response.text
    assert "SENSITIVE_SOURCE_PAYLOAD" not in response.text


def test_missing_openai_key_never_returns_legacy_stub(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    stub = PrioritizeTasksResponse(requestId="request", results=[])
    with pytest.raises(InferenceFailure) as captured:
        call_structured("prompt", "payload", PrioritizeTasksResponse, stub)
    assert captured.value.code == InferenceFailureCode.UNAVAILABLE
    get_settings.cache_clear()


@pytest.mark.parametrize(
    "sdk_error",
    [
        ContentFilterFinishReasonError(),
        LengthFinishReasonError(completion=MagicMock()),
    ],
)
def test_finish_reason_errors_map_to_safe_502(
    foundation_client, monkeypatch, sdk_error
):
    monkeypatch.setenv("OPENAI_API_KEY", "SENSITIVE_API_KEY")
    get_settings.cache_clear()
    parse = MagicMock(side_effect=sdk_error)
    fake_client = SimpleNamespace(
        beta=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
        )
    )
    monkeypatch.setattr(llm_module, "OpenAI", lambda **_kwargs: fake_client)

    response = foundation_client.get(
        "/internal/v1/__foundation__/llm",
        headers={"X-Internal-Token": "foundation-token"},
    )
    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "code": "AI_UPSTREAM_INVALID_RESPONSE",
            "message": "AI inference failed.",
        }
    }
    assert "SENSITIVE_API_KEY" not in response.text
    assert "SENSITIVE_SOURCE_PAYLOAD" not in response.text


def test_cors_wildcard_is_not_enabled(foundation_client):
    response = TestClient(app).options(
        "/health",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") != "*"
