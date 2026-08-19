import asyncio
import hashlib
from io import BytesIO
import json

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.routers.schedule_ocr import get_schedule_ocr_service
from app.schedule_ocr.body_limit import ScheduleOcrBodyLimitMiddleware
from app.schedule_ocr.engine import OcrCandidate
from app.schedule_ocr.service import ScheduleOcrService
from main import app


class FakeEngine:
    def recognize(self, _cell: Image.Image, *, timeout_seconds: float | None = None) -> OcrCandidate:
        return OcrCandidate("D", 0.95)


def fake_service() -> ScheduleOcrService:
    return ScheduleOcrService(
        FakeEngine(), max_image_bytes=10 * 1024 * 1024, min_image_width=640,
        min_image_height=480, max_image_pixels=16_000_000, review_threshold=0.85,
    )


def synthetic_png() -> bytes:
    image = Image.new("RGB", (1600, 1200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 56, 132, 104), fill="black")
    for column in range(32):
        x = round(160 + column * (1536 - 160) / 31)
        draw.line((x, 300, x, 1040), fill="black", width=5)
    for row in range(17):
        y = round(300 + row * (1040 - 300) / 16)
        draw.line((160, y, 1536, y), fill="black", width=5)
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def request_parts(row_index: str = "3") -> tuple[dict, dict]:
    image = synthetic_png()
    return (
        {"image": ("synthetic.png", image, "image/png")},
        {
            "yearMonth": "2026-02", "templateId": "NURSE_HAND_FIXED_V1", "rowIndex": row_index,
            "expectedWidth": "1600", "expectedHeight": "1200",
            "expectedSha256": hashlib.sha256(image).hexdigest(),
        },
    )


def test_main_app_actual_post_auth_success_and_body_coercion(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_API_TOKEN", "test-token")
    app.dependency_overrides[get_schedule_ocr_service] = fake_service
    try:
        client = TestClient(app)
        files, data = request_parts()
        success = client.post(
            "/internal/v1/schedules/ocr", headers={"X-Internal-Token": "test-token"},
            files=files, data=data,
        )
        assert success.status_code == 200
        assert success.json()["cells"][0]["token"] == "DAY"

        files, data = request_parts("+3")
        invalid = client.post(
            "/internal/v1/schedules/ocr", headers={"X-Internal-Token": "test-token"},
            files=files, data=data,
        )
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "SCHEDULE_OCR_INVALID_REQUEST"
    finally:
        app.dependency_overrides.pop(get_schedule_ocr_service, None)


def test_main_app_auth_errors_are_stable_and_hide_setting_name(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setenv("INTERNAL_API_TOKEN", "test-token")
    unauthorized = client.post("/internal/v1/schedules/ocr")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "SCHEDULE_OCR_UNAUTHORIZED"

    monkeypatch.delenv("INTERNAL_API_TOKEN")
    unavailable = client.post("/internal/v1/schedules/ocr")
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "SCHEDULE_OCR_AUTH_UNAVAILABLE"
    assert "INTERNAL_API_TOKEN" not in unavailable.text


def test_route_retains_image_byte_limit_below_multipart_limit(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_API_TOKEN", "test-token")
    app.dependency_overrides[get_schedule_ocr_service] = fake_service
    oversized_image = b"\x89PNG\r\n\x1a\n" + b"0" * (10 * 1024 * 1024)
    try:
        response = TestClient(app).post(
            "/internal/v1/schedules/ocr", headers={"X-Internal-Token": "test-token"},
            files={"image": ("oversized.png", oversized_image, "image/png")},
            data={
                "yearMonth": "2026-02", "templateId": "NURSE_HAND_FIXED_V1", "rowIndex": "3",
                "expectedWidth": "1600", "expectedHeight": "1200",
                "expectedSha256": hashlib.sha256(oversized_image).hexdigest(),
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "SCHEDULE_OCR_INVALID_REQUEST"
    finally:
        app.dependency_overrides.pop(get_schedule_ocr_service, None)


def test_streaming_body_limit_counts_chunks_without_content_length() -> None:
    messages = [
        {"type": "http.request", "body": b"1234", "more_body": True},
        {"type": "http.request", "body": b"5678", "more_body": False},
    ]
    sent: list[dict] = []
    downstream_completed = False

    async def receive() -> dict:
        return messages.pop(0)

    async def send(message: dict) -> None:
        sent.append(message)

    async def downstream(_scope, limited_receive, _send) -> None:
        nonlocal downstream_completed
        while True:
            message = await limited_receive()
            if not message.get("more_body"):
                break
        downstream_completed = True

    scope = {
        "type": "http", "method": "POST", "path": "/internal/v1/schedules/ocr", "headers": [],
    }
    asyncio.run(ScheduleOcrBodyLimitMiddleware(downstream, max_body_bytes=7)(scope, receive, send))

    assert downstream_completed is False
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"]["code"] == "SCHEDULE_OCR_BODY_TOO_LARGE"
