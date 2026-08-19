import json
from collections.abc import Awaitable, Callable
from typing import Any

SCHEDULE_OCR_PATH = "/internal/v1/schedules/ocr"
DEFAULT_MAX_MULTIPART_BYTES = 11 * 1024 * 1024


class _BodyLimitExceeded(Exception):
    pass


class ScheduleOcrBodyLimitMiddleware:
    def __init__(self, app: Callable[..., Awaitable[None]], max_body_bytes: int = DEFAULT_MAX_MULTIPART_BYTES) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Awaitable[dict]], send: Callable[..., Awaitable[None]]) -> None:
        if scope.get("type") != "http" or scope.get("path") != SCHEDULE_OCR_PATH or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        content_length = next(
            (value for name, value in scope.get("headers", []) if name.lower() == b"content-length"),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_bytes:
                    await self._send_rejection(send)
                    return
            except ValueError:
                pass

        received = 0
        response_started = False

        async def limited_receive() -> dict:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _BodyLimitExceeded
            return message

        async def tracked_send(message: dict) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _BodyLimitExceeded:
            if response_started:
                raise
            await self._send_rejection(send)

    @staticmethod
    async def _send_rejection(send: Callable[..., Awaitable[None]]) -> None:
        payload = json.dumps(
            {
                "error": {
                    "code": "SCHEDULE_OCR_BODY_TOO_LARGE",
                    "message": "multipart 요청 크기가 최대 기준을 초과합니다.",
                }
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json; charset=utf-8"), (b"content-length", str(len(payload)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": payload})
