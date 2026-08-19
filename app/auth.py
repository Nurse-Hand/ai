import secrets

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import Settings, get_settings

_internal_token_header = APIKeyHeader(name="X-Internal-Token", auto_error=False)


async def verify_internal_token(
    x_internal_token: str | None = Security(_internal_token_header),
    settings: Settings = Depends(get_settings),
) -> None:
    expected = settings.internal_token
    if not expected:
        raise HTTPException(status_code=503, detail={"code": "INTERNAL_AUTH_UNAVAILABLE"})
    if x_internal_token is None or not secrets.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=401, detail={"code": "INVALID_INTERNAL_TOKEN"})
