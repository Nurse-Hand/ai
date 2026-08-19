import json

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
