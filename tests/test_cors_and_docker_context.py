from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
import pytest

from app.cors import ALLOWED_CORS_HEADERS, ALLOWED_CORS_METHODS, parse_cors_allowed_origins
from app.routers.speakers import router as speakers_router
from main import app

ROOT = Path(__file__).resolve().parents[1]


def test_cors_is_default_deny_without_breaking_legacy_same_origin_paths() -> None:
    client = TestClient(app)
    response = client.get("/health", headers={"Origin": "https://untrusted.example"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_explicit_cors_origin_allows_only_configured_origin() -> None:
    probe = FastAPI()
    probe.add_middleware(
        CORSMiddleware,
        allow_origins=parse_cors_allowed_origins("https://dashboard.example"),
        allow_methods=ALLOWED_CORS_METHODS,
        allow_headers=ALLOWED_CORS_HEADERS,
    )

    @probe.get("/probe")
    def read_probe() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(probe)
    allowed = client.get("/probe", headers={"Origin": "https://dashboard.example"})
    denied = client.get("/probe", headers={"Origin": "https://untrusted.example"})
    assert allowed.headers["access-control-allow-origin"] == "https://dashboard.example"
    assert "access-control-allow-origin" not in denied.headers


def test_speaker_delete_route_preflight_is_allowed_for_configured_origin() -> None:
    probe = FastAPI()
    probe.include_router(speakers_router)
    probe.add_middleware(
        CORSMiddleware,
        allow_origins=["https://dashboard.example"],
        allow_methods=ALLOWED_CORS_METHODS,
        allow_headers=ALLOWED_CORS_HEADERS,
    )
    response = TestClient(probe).options(
        "/api/speakers/speaker-1",
        headers={
            "Origin": "https://dashboard.example",
            "Access-Control-Request-Method": "DELETE",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://dashboard.example"
    assert "DELETE" in response.headers["access-control-allow-methods"]


@pytest.mark.parametrize("raw", ["*", "null", "file://dashboard", "https://ok.example,*"])
def test_unsafe_cors_values_fail_closed(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_cors_allowed_origins(raw)


def test_docker_context_excludes_secrets_runtime_data_and_test_artifacts() -> None:
    patterns = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for required in {
        ".git", ".env", ".env.*", ".venv", "data", "tmp", ".pytest_cache",
        "**/__pycache__", "tests", "*.png", "*.jpg", "*.wav",
    }:
        assert required in patterns
