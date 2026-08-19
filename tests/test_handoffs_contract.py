from app.routers import handoffs as handoff_router

REQUEST_ID = "00000000-0000-4000-8000-000000000011"
PATIENT_ID = "00000000-0000-4000-8000-000000000012"
EVENT_ID = "00000000-0000-4000-8000-000000000013"
TASK_ID = "00000000-0000-4000-8000-000000000014"
ITEM_ID = "00000000-0000-4000-8000-000000000015"
SECTIONS = ["PATIENT_STATUS", "PAIN", "TREATMENT", "DIET", "ACTIVITY", "OBSERVATION"]


def patients():
    return [
        {
            "patientId": PATIENT_ID,
            "timelineEvents": [
                {
                    "id": EVENT_ID,
                    "occurredAt": "2026-08-19T00:00:00Z",
                    "type": "OBSERVATION",
                    "summary": "synthetic observation",
                    "sourceReference": "synthetic-ref",
                }
            ],
            "tasks": [
                {
                    "id": TASK_ID,
                    "title": "synthetic task",
                    "dueAt": None,
                    "effectivePriority": "NORMAL",
                    "version": 1,
                    "sourceReferences": ["synthetic-task-ref"],
                }
            ],
        }
    ]


def evidence():
    return {"sourceType": "TIMELINE_EVENT", "sourceId": EVENT_ID, "patientId": PATIENT_ID}


def model_result(model, payload):
    return model.model_validate(payload)


def precheck_response():
    return {
        "requestId": REQUEST_ID,
        "modelVersion": "test-model-v1",
        "contractVersion": "handoff-precheck-v1",
        "generatedAt": "2026-08-19T00:00:01Z",
        "questions": [
            {
                "questionKey": "question-1",
                "patientId": PATIENT_ID,
                "severity": "RECOMMENDED",
                "prompt": "Include this observation?",
                "reason": "It may be relevant.",
                "evidence": [evidence()],
            }
        ],
    }


def generate_body():
    return {
        "requestId": REQUEST_ID,
        "templateId": "NURSING_HANDOFF_V1",
        "includeUnverified": True,
        "patients": patients(),
        "precheckItems": [
            {
                "id": ITEM_ID,
                "severity": "RECOMMENDED",
                "question": "Include this observation?",
                "answer": "UNVERIFIED",
                "evidence": [evidence()],
            }
        ],
    }


def generate_response():
    return {
        "requestId": REQUEST_ID,
        "modelVersion": "test-model-v1",
        "contractVersion": "handoff-draft-v1",
        "generatedAt": "2026-08-19T00:00:02Z",
        "patients": [
            {
                "patientId": PATIENT_ID,
                "sections": [
                    {
                        "section": section,
                        "content": f"Synthetic {section} draft.",
                        "citations": [evidence()] if section == "OBSERVATION" else [],
                    }
                    for section in SECTIONS
                ],
            }
        ],
        "warnings": [
            {
                "code": "UNVERIFIED_INFORMATION",
                "itemId": ITEM_ID,
                "patientId": PATIENT_ID,
                "message": "This information remains unverified.",
                "evidence": [evidence()],
            }
        ],
    }


def test_precheck_then_generate_return_strict_200(client, auth_headers, monkeypatch):
    def fake(_prompt, _content, model):
        payload = precheck_response() if model.__name__ == "HandoffPrecheckResponse" else generate_response()
        return model_result(model, payload)

    monkeypatch.setattr(handoff_router, "call_structured", fake)
    precheck = client.post(
        "/internal/v1/handoffs/precheck",
        headers=auth_headers,
        json={"requestId": REQUEST_ID, "patients": patients()},
    )
    assert precheck.status_code == 200
    assert precheck.json()["questions"][0]["severity"] == "RECOMMENDED"

    generated = client.post(
        "/internal/v1/handoffs/generate", headers=auth_headers, json=generate_body()
    )
    assert generated.status_code == 200
    assert {item["section"] for item in generated.json()["patients"][0]["sections"]} == set(SECTIONS)
    assert generated.json()["warnings"][0]["code"] == "UNVERIFIED_INFORMATION"


def test_handoff_endpoints_require_auth(client):
    assert client.post(
        "/internal/v1/handoffs/precheck", json={"requestId": REQUEST_ID, "patients": patients()}
    ).status_code == 401
    assert client.post("/internal/v1/handoffs/generate", json=generate_body()).status_code == 401


def test_invalid_template_and_severity_are_rejected(client, auth_headers):
    body = generate_body()
    body["templateId"] = "SBAR"
    assert client.post("/internal/v1/handoffs/generate", headers=auth_headers, json=body).status_code == 422

    body = {"requestId": REQUEST_ID, "patients": patients(), "unexpected": True}
    assert client.post("/internal/v1/handoffs/precheck", headers=auth_headers, json=body).status_code == 422


def test_handoff_scalar_coercion_and_untrimmed_text_are_rejected(client, auth_headers):
    body = generate_body()
    body["includeUnverified"] = "false"
    assert client.post("/internal/v1/handoffs/generate", headers=auth_headers, json=body).status_code == 422

    body = generate_body()
    body["patients"][0]["tasks"][0]["version"] = "1"
    assert client.post("/internal/v1/handoffs/generate", headers=auth_headers, json=body).status_code == 422

    body = {"requestId": REQUEST_ID, "patients": patients()}
    body["patients"][0]["timelineEvents"][0]["summary"] = " leading"
    assert client.post("/internal/v1/handoffs/precheck", headers=auth_headers, json=body).status_code == 422

    body["patients"][0]["timelineEvents"][0]["summary"] = "control\ttext"
    assert client.post("/internal/v1/handoffs/precheck", headers=auth_headers, json=body).status_code == 422


def test_unknown_handoff_evidence_is_rejected(client, auth_headers, monkeypatch):
    payload = precheck_response()
    payload["questions"][0]["evidence"][0]["sourceId"] = "00000000-0000-4000-8000-000000000099"
    monkeypatch.setattr(
        handoff_router,
        "call_structured",
        lambda _prompt, _content, model: model_result(model, payload),
    )
    response = client.post(
        "/internal/v1/handoffs/precheck",
        headers=auth_headers,
        json={"requestId": REQUEST_ID, "patients": patients()},
    )
    assert response.status_code == 502


def test_duplicate_section_result_is_rejected(client, auth_headers, monkeypatch):
    payload = generate_response()
    payload["patients"][0]["sections"][-1]["section"] = "PAIN"
    monkeypatch.setattr(
        handoff_router,
        "call_structured",
        lambda _prompt, _content, model: model_result(model, payload),
    )
    response = client.post(
        "/internal/v1/handoffs/generate", headers=auth_headers, json=generate_body()
    )
    assert response.status_code == 502


def test_duplicate_warning_evidence_is_rejected(client, auth_headers, monkeypatch):
    payload = generate_response()
    payload["warnings"][0]["evidence"].append(evidence())
    result = model_result(handoff_router.GenerateHandoffResponse, payload)
    monkeypatch.setattr(handoff_router, "call_structured", lambda *_args: result)
    response = client.post(
        "/internal/v1/handoffs/generate", headers=auth_headers, json=generate_body()
    )
    assert response.status_code == 502
