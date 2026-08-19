from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.routers.schedule_ocr import get_schedule_ocr_service, router, schedule_ocr_error_handler
from app.schedule_ocr.engine import OcrCandidate
from app.schedule_ocr.errors import ScheduleOcrError
from app.schedule_ocr.service import ScheduleOcrService
from scripts.generate_schedule_ocr_openapi import contract_schema

ROOT = Path(__file__).resolve().parents[1]


class FakeEngine:
    def recognize(self, _cell: Image.Image) -> OcrCandidate:
        return OcrCandidate("D", 0.95)


class UnavailableEngine:
    def recognize(self, _cell: Image.Image) -> OcrCandidate:
        raise ScheduleOcrError("SCHEDULE_OCR_ENGINE_UNAVAILABLE", "OCR 엔진을 사용할 수 없습니다.", 503)


def synthetic_png() -> bytes:
    width, height = 1600, 1200
    grid_left, grid_top, grid_right, grid_bottom = 160, 300, 1536, 1040
    row_count, column_count = 16, 31
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 56, 132, 104), fill="black")
    for column in range(column_count + 1):
        x = round(grid_left + column * (grid_right - grid_left) / column_count)
        draw.line((x, grid_top, x, grid_bottom), fill="black", width=5)
    for row in range(row_count + 1):
        y = round(grid_top + row * (grid_bottom - grid_top) / row_count)
        draw.line((grid_left, y, grid_right, y), fill="black", width=5)
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def fake_service() -> ScheduleOcrService:
    return ScheduleOcrService(
        FakeEngine(), max_image_bytes=10 * 1024 * 1024, min_image_width=640,
        min_image_height=480, max_image_pixels=16_000_000, review_threshold=0.85,
    )


def unavailable_service() -> ScheduleOcrService:
    return ScheduleOcrService(
        UnavailableEngine(), max_image_bytes=10 * 1024 * 1024, min_image_width=640,
        min_image_height=480, max_image_pixels=16_000_000, review_threshold=0.85,
    )


def client() -> TestClient:
    os.environ["INTERNAL_API_TOKEN"] = "test-token"
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ScheduleOcrError, schedule_ocr_error_handler)
    app.dependency_overrides[get_schedule_ocr_service] = fake_service
    return TestClient(app)


def valid_request(test_client: TestClient):
    image = synthetic_png()
    return test_client.post(
        "/internal/v1/schedules/ocr",
        headers={"X-Internal-Token": "test-token"},
        files={"image": ("synthetic.png", image, "image/png")},
        data={
            "yearMonth": "2026-02", "templateId": "NURSE_HAND_FIXED_V1", "rowIndex": "3",
            "expectedWidth": "1600", "expectedHeight": "1200",
            "expectedSha256": hashlib.sha256(image).hexdigest(),
        },
    )


def test_auth_is_required() -> None:
    response = client().post("/internal/v1/schedules/ocr")
    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "code": "SCHEDULE_OCR_UNAUTHORIZED",
            "message": "유효한 내부 인증 token이 필요합니다.",
        }
    }


def test_success_wire_contract() -> None:
    response = valid_request(client())
    assert response.status_code == 200
    payload = response.json()
    assert payload["contractVersion"] == "schedule-ocr.v1"
    assert payload["templateId"] == "NURSE_HAND_FIXED_V1"
    assert payload["yearMonth"] == "2026-02"
    assert payload["cells"][0] == {
        "date": "2026-02-01", "token": "DAY", "confidence": 0.95, "needsReview": False,
    }
    assert len(payload["cells"]) == 28


def test_missing_form_has_stable_error_envelope() -> None:
    response = client().post(
        "/internal/v1/schedules/ocr", headers={"X-Internal-Token": "test-token"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "SCHEDULE_OCR_INVALID_REQUEST"


def test_invalid_row_index_has_stable_error_envelope() -> None:
    image = synthetic_png()
    response = client().post(
        "/internal/v1/schedules/ocr",
        headers={"X-Internal-Token": "test-token"},
        files={"image": ("synthetic.png", image, "image/png")},
        data={
            "yearMonth": "2026-02", "templateId": "NURSE_HAND_FIXED_V1", "rowIndex": "1.5",
            "expectedWidth": "1600", "expectedHeight": "1200",
            "expectedSha256": hashlib.sha256(image).hexdigest(),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "SCHEDULE_OCR_INVALID_REQUEST"


def test_noncanonical_row_index_and_unknown_field_are_rejected() -> None:
    image = synthetic_png()
    fields = {
        "yearMonth": "2026-02", "templateId": "NURSE_HAND_FIXED_V1", "rowIndex": "+3",
        "expectedWidth": "1600", "expectedHeight": "1200",
        "expectedSha256": hashlib.sha256(image).hexdigest(),
    }
    test_client = client()
    noncanonical = test_client.post(
        "/internal/v1/schedules/ocr", headers={"X-Internal-Token": "test-token"},
        files={"image": ("synthetic.png", image, "image/png")}, data=fields,
    )
    assert noncanonical.status_code == 400

    fields["rowIndex"] = "3"
    fields["unexpected"] = "not-allowed"
    unknown = test_client.post(
        "/internal/v1/schedules/ocr", headers={"X-Internal-Token": "test-token"},
        files={"image": ("synthetic.png", image, "image/png")}, data=fields,
    )
    assert unknown.status_code == 400
    assert unknown.json()["error"]["code"] == "SCHEDULE_OCR_INVALID_REQUEST"


def test_expected_image_metadata_is_enforced() -> None:
    image = synthetic_png()
    response = client().post(
        "/internal/v1/schedules/ocr", headers={"X-Internal-Token": "test-token"},
        files={"image": ("synthetic.png", image, "image/png")},
        data={
            "yearMonth": "2026-02", "templateId": "NURSE_HAND_FIXED_V1", "rowIndex": "3",
            "expectedWidth": "1599", "expectedHeight": "1200", "expectedSha256": "0" * 64,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "SCHEDULE_OCR_INVALID_REQUEST"


def test_engine_unavailable_has_stable_error_envelope() -> None:
    test_client = client()
    test_client.app.dependency_overrides[get_schedule_ocr_service] = unavailable_service
    response = valid_request(test_client)
    assert response.status_code == 503
    assert response.json() == {
        "error": {"code": "SCHEDULE_OCR_ENGINE_UNAVAILABLE", "message": "OCR 엔진을 사용할 수 없습니다."}
    }


def test_openapi_declares_required_multipart_and_errors() -> None:
    schema = client().get("/openapi.json").json()
    operation = schema["paths"]["/internal/v1/schedules/ocr"]["post"]
    assert operation["parameters"] == [
        {
            "name": "X-Internal-Token", "in": "header", "required": True,
            "schema": {"type": "string"}, "description": "Internal service authentication token.",
        }
    ]
    body_schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert body_schema["required"] == [
        "image", "yearMonth", "templateId", "rowIndex",
        "expectedWidth", "expectedHeight", "expectedSha256",
    ]
    assert body_schema["additionalProperties"] is False
    assert body_schema["properties"]["rowIndex"]["type"] == "integer"
    response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema["$ref"] == "#/components/schemas/ScheduleOcrResponse"
    assert schema["components"]["schemas"]["ScheduleOcrCell"]["properties"]["token"]["enum"] == [
        "DAY", "EVENING", "NIGHT", "OFF", "UNKNOWN",
    ]
    assert set(operation["responses"]) >= {"200", "400", "401", "413", "415", "422", "502", "503", "504"}


def test_generated_openapi_has_no_drift() -> None:
    artifact = json.loads((ROOT / "openapi" / "schedule-ocr.v1.json").read_text(encoding="utf-8"))
    assert artifact == contract_schema()


def test_openapi_generator_is_directly_executable() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_schedule_ocr_openapi.py")],
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
