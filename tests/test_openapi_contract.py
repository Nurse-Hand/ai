import json

from app.errors import ErrorResponse
from scripts.generate_openapi import ARTIFACT, INTERNAL_PATHS, render_openapi


def test_versioned_openapi_has_no_drift():
    assert ARTIFACT.read_text(encoding="utf-8") == render_openapi()


def test_internal_openapi_contains_exactly_five_authenticated_200_contracts():
    schema = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert set(schema["paths"]) == INTERNAL_PATHS
    assert schema["components"]["securitySchemes"]["APIKeyHeader"] == {
        "type": "apiKey",
        "in": "header",
        "name": "X-Internal-Token",
    }
    for path in INTERNAL_PATHS:
        operation = schema["paths"][path]["post"]
        assert "200" in operation["responses"]
        assert "201" not in operation["responses"]
        assert operation["security"] == [{"APIKeyHeader": []}]
        for status in ("401", "422", "429", "502", "503", "504"):
            response_schema = operation["responses"][status]["content"]["application/json"]["schema"]
            assert response_schema == {"$ref": "#/components/schemas/ErrorResponse"}

    components = schema["components"]["schemas"]
    assert "ErrorResponse" in components
    assert "HTTPValidationError" not in components
    assert "ValidationError" not in components
    assert all("SpeakerRecord" not in name for name in components)


def test_runtime_validation_envelope_matches_documented_error_schema(client, auth_headers):
    response = client.post(
        "/internal/v1/tasks/extract",
        headers=auth_headers,
        json={"requestId": "not-a-uuid", "evidence": []},
    )
    assert response.status_code == 422
    parsed = ErrorResponse.model_validate(response.json())
    assert parsed.error.code == "INVALID_INPUT"
    assert set(response.json()) == {"error"}
    assert set(response.json()["error"]) == {"code", "message"}
