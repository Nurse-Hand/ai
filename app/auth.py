import os
import secrets
from dataclasses import dataclass

from fastapi import Header, HTTPException


@dataclass(frozen=True, slots=True)
class InternalAuthError(Exception):
    status_code: int


def require_internal_token(candidate: str | None) -> None:
    expected = os.getenv("INTERNAL_API_TOKEN")
    if not expected:
        raise InternalAuthError(503)
    if candidate is None or not secrets.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8")):
        raise InternalAuthError(401)


async def verify_internal_token(x_internal_token: str | None = Header(None)) -> None:
    """백엔드->AI 서버 내부 호출 인증. Header: X-Internal-Token: {serviceToken} (노션 확정)"""
    try:
        require_internal_token(x_internal_token)
    except InternalAuthError as exc:
        message = "내부 인증을 사용할 수 없습니다." if exc.status_code == 503 else "내부 인증에 실패했습니다."
        raise HTTPException(status_code=exc.status_code, detail=message) from exc
