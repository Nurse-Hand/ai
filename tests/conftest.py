import json

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("INTERNAL_TOKEN", "test-internal-token")
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest.fixture
def auth_headers():
    return {"X-Internal-Token": "test-internal-token"}


def validated(model, payload):
    return model.model_validate_json(json.dumps(payload))
