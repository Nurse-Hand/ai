import pytest

from app.errors import InferenceFailure, InferenceFailureCode
from app.routers import tasks as task_router

REQUEST_ID = "00000000-0000-4000-8000-000000000001"
PATIENT_ID = "00000000-0000-4000-8000-000000000002"
SOURCE_ID = "00000000-0000-4000-8000-000000000003"
RECORD_ID = "00000000-0000-4000-8000-000000000004"


def extract_body():
    return {
        "requestId": REQUEST_ID,
        "evidence": [
            {
                "recordId": RECORD_ID,
                "sourceType": "TIMELINE_EVENT",
                "sourceId": SOURCE_ID,
                "patientId": PATIENT_ID,
                "workDate": "2026-08-19T00:00:00Z",
                "summary": "synthetic follow-up evidence",
            }
        ],
    }


def candidate():
    return {
        "candidateKey": "candidate-1",
        "patientId": PATIENT_ID,
        "title": "synthetic follow-up task",
        "description": None,
        "dueAt": None,
        "evidenceSourceIds": [SOURCE_ID],
        "confidence": "HIGH",
    }


def model_result(model, payload):
    return model.model_validate(payload)


def test_extract_and_prioritize_return_strict_200(client, auth_headers, monkeypatch):
    def fake(_prompt, _content, model):
        if model.__name__ == "ExtractTasksResponse":
            return model_result(model, {"requestId": REQUEST_ID, "candidates": [candidate()]})
        return model_result(
            model,
            {
                "requestId": REQUEST_ID,
                "suggestions": [
                    {
                        "candidateKey": "candidate-1",
                        "suggestedPriority": "HIGH",
                        "reasons": ["requires timely follow-up"],
                        "confidence": "MEDIUM",
                        "evidenceSourceIds": [SOURCE_ID],
                    }
                ],
            },
        )

    monkeypatch.setattr(task_router, "call_structured", fake)
    extracted = client.post("/internal/v1/tasks/extract", headers=auth_headers, json=extract_body())
    assert extracted.status_code == 200
    assert extracted.json()["candidates"][0]["confidence"] == "HIGH"

    prioritized = client.post(
        "/internal/v1/tasks/prioritize",
        headers=auth_headers,
        json={"requestId": REQUEST_ID, "candidates": [candidate()]},
    )
    assert prioritized.status_code == 200
    suggestion = prioritized.json()["suggestions"][0]
    assert suggestion["suggestedPriority"] == "HIGH"
    assert "score" not in suggestion
    assert "priority" not in suggestion


@pytest.mark.parametrize(
    "path,body",
    [
        ("/internal/v1/tasks/extract", extract_body()),
        (
            "/internal/v1/tasks/prioritize",
            {"requestId": REQUEST_ID, "candidates": [candidate()]},
        ),
    ],
)
def test_task_endpoints_require_auth(client, path, body):
    assert client.post(path, json=body).status_code == 401


def test_unknown_request_field_and_invalid_enum_are_rejected(client, auth_headers):
    body = extract_body()
    body["unexpected"] = True
    assert client.post("/internal/v1/tasks/extract", headers=auth_headers, json=body).status_code == 422

    body = {"requestId": REQUEST_ID, "candidates": [candidate()]}
    body["candidates"][0]["confidence"] = "CERTAIN"
    assert client.post("/internal/v1/tasks/prioritize", headers=auth_headers, json=body).status_code == 422


def test_task_text_whitespace_and_control_characters_are_rejected(client, auth_headers):
    for summary in (" leading", "trailing ", "control\ntext"):
        body = extract_body()
        body["evidence"][0]["summary"] = summary
        response = client.post("/internal/v1/tasks/extract", headers=auth_headers, json=body)
        assert response.status_code == 422


def test_unknown_evidence_id_from_model_is_rejected(client, auth_headers, monkeypatch):
    unknown = "00000000-0000-4000-8000-000000000099"

    def fake(_prompt, _content, model):
        value = candidate()
        value["evidenceSourceIds"] = [unknown]
        return model_result(model, {"requestId": REQUEST_ID, "candidates": [value]})

    monkeypatch.setattr(task_router, "call_structured", fake)
    response = client.post("/internal/v1/tasks/extract", headers=auth_headers, json=extract_body())
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "AI_UPSTREAM_INVALID_RESPONSE"


def test_partial_prioritize_result_is_rejected(client, auth_headers, monkeypatch):
    monkeypatch.setattr(
        task_router,
        "call_structured",
        lambda _prompt, _content, model: model_result(
            model, {"requestId": REQUEST_ID, "suggestions": []}
        ),
    )
    response = client.post(
        "/internal/v1/tasks/prioritize",
        headers=auth_headers,
        json={"requestId": REQUEST_ID, "candidates": [candidate()]},
    )
    assert response.status_code == 502


@pytest.mark.parametrize(
    "code,status",
    [
        (InferenceFailureCode.TIMEOUT, 504),
        (InferenceFailureCode.RATE_LIMITED, 429),
        (InferenceFailureCode.UNAVAILABLE, 503),
        (InferenceFailureCode.INVALID_RESPONSE, 502),
    ],
)
def test_upstream_errors_are_stably_mapped(client, auth_headers, monkeypatch, code, status):
    def fail(*_args):
        raise InferenceFailure(code)

    monkeypatch.setattr(task_router, "call_structured", fail)
    response = client.post("/internal/v1/tasks/extract", headers=auth_headers, json=extract_body())
    assert response.status_code == status
    assert response.json()["error"]["code"] == code.value


def test_secret_and_source_payload_are_not_logged(client, auth_headers, monkeypatch, caplog):
    monkeypatch.setattr(
        task_router,
        "call_structured",
        lambda _prompt, _content, _model: (_ for _ in ()).throw(
            InferenceFailure(InferenceFailureCode.UNAVAILABLE)
        ),
    )
    body = extract_body()
    body["evidence"][0]["summary"] = "SENSITIVE_SYNTHETIC_PAYLOAD"
    client.post("/internal/v1/tasks/extract", headers=auth_headers, json=body)
    rendered = caplog.text
    assert "test-internal-token" not in rendered
    assert "SENSITIVE_SYNTHETIC_PAYLOAD" not in rendered
