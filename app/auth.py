import secrets
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings

_internal_token_header = APIKeyHeader(name="X-Internal-Token", auto_error=False)


@dataclass(frozen=True, slots=True)
class InternalAuthError(Exception):
    status_code: int


def require_internal_token(candidate: str | None, expected: str | None) -> None:
    if not expected:
        raise InternalAuthError(503)
    if candidate is None or not secrets.compare_digest(candidate, expected):
        raise InternalAuthError(401)


async def verify_internal_token(
    x_internal_token: str | None = Security(_internal_token_header),
    settings: Settings = Depends(get_settings),
) -> None:
    try:
        require_internal_token(x_internal_token, settings.internal_token)
    except InternalAuthError as exc:
        code = "INTERNAL_AUTH_UNAVAILABLE" if exc.status_code == 503 else "INVALID_INTERNAL_TOKEN"
        raise HTTPException(status_code=exc.status_code, detail={"code": code}) from exc
