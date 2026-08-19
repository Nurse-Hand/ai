import os

from fastapi import Header, HTTPException


async def verify_internal_token(x_internal_token: str | None = Header(None)) -> None:
    """백엔드->AI 서버 내부 호출 인증. Header: X-Internal-Token: {serviceToken} (노션 확정)"""
    expected = os.getenv("INTERNAL_API_TOKEN")
    if not expected:
        raise HTTPException(status_code=500, detail="INTERNAL_API_TOKEN 미설정")

    if x_internal_token != expected:
        raise HTTPException(status_code=401, detail="인증 실패: 유효하지 않은 토큰")
