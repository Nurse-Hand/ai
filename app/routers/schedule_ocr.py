from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse

from app.auth import InternalAuthError, require_internal_token
from app.schedule_ocr.engine import TesseractCellOcrEngine
from app.schedule_ocr.errors import ScheduleOcrError
from app.schedule_ocr.multipart import ScheduleOcrMultipartRequest, parse_schedule_ocr_form
from app.schedule_ocr.schemas import ScheduleOcrErrorResponse, ScheduleOcrResponse
from app.schedule_ocr.service import ScheduleOcrService
from app.schedule_ocr.settings import ScheduleOcrSettings, get_schedule_ocr_settings

ERROR_EXAMPLE = {"error": {"code": "SCHEDULE_OCR_INVALID_REQUEST", "message": "요청을 처리할 수 없습니다."}}
MULTIPART_SCHEMA = {
    "parameters": [
        {
            "name": "X-Internal-Token",
            "in": "header",
            "required": True,
            "schema": {"type": "string"},
            "description": "Internal service authentication token.",
        }
    ],
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "image", "yearMonth", "templateId", "rowIndex",
                        "expectedWidth", "expectedHeight", "expectedSha256",
                    ],
                    "properties": {
                        "image": {"type": "string", "format": "binary"},
                        "yearMonth": {"type": "string", "pattern": "^20[0-9]{2}-(0[1-9]|1[0-2])$", "example": "2026-08"},
                        "templateId": {"type": "string", "enum": ["NURSE_HAND_FIXED_V1"]},
                        "rowIndex": {"type": "integer", "minimum": 0, "maximum": 15, "example": 3},
                        "expectedWidth": {"type": "integer", "minimum": 1, "example": 1600},
                        "expectedHeight": {"type": "integer", "minimum": 1, "example": 1200},
                        "expectedSha256": {
                            "type": "string", "pattern": "^[0-9a-f]{64}$",
                            "example": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                        },
                    },
                }
            }
        },
    }
}


async def verify_schedule_ocr_token(
    x_internal_token: Annotated[
        str | None,
        Header(alias="X-Internal-Token", include_in_schema=False),
    ] = None,
) -> None:
    try:
        require_internal_token(x_internal_token)
    except InternalAuthError as exc:
        if exc.status_code == 503:
            raise ScheduleOcrError(
                "SCHEDULE_OCR_AUTH_UNAVAILABLE", "내부 인증을 사용할 수 없습니다.", 503,
            ) from exc
        raise ScheduleOcrError(
            "SCHEDULE_OCR_UNAUTHORIZED", "유효한 내부 인증 token이 필요합니다.", 401,
        ) from exc


router = APIRouter(
    prefix="/internal/v1/schedules",
    tags=["schedule-ocr"],
    dependencies=[Depends(verify_schedule_ocr_token)],
)


def get_schedule_ocr_service(
    settings: ScheduleOcrSettings = Depends(get_schedule_ocr_settings),
) -> ScheduleOcrService:
    return ScheduleOcrService(
        TesseractCellOcrEngine(
            settings.tesseract_bin,
            settings.tesseract_language,
            settings.schedule_ocr_confidence_threshold,
            settings.schedule_ocr_timeout_seconds,
        ),
        max_image_bytes=settings.schedule_ocr_max_image_bytes,
        min_image_width=settings.schedule_ocr_min_image_width,
        min_image_height=settings.schedule_ocr_min_image_height,
        max_image_pixels=settings.schedule_ocr_max_image_pixels,
        review_threshold=settings.schedule_ocr_confidence_threshold,
    )


async def schedule_ocr_error_handler(_request: Request, exc: ScheduleOcrError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})


@router.post(
    "/ocr",
    response_model=ScheduleOcrResponse,
    status_code=200,
    openapi_extra=MULTIPART_SCHEMA,
    responses={
        401: {"model": ScheduleOcrErrorResponse},
        400: {"model": ScheduleOcrErrorResponse, "content": {"application/json": {"example": ERROR_EXAMPLE}}},
        413: {"model": ScheduleOcrErrorResponse},
        415: {"model": ScheduleOcrErrorResponse},
        422: {"model": ScheduleOcrErrorResponse},
        502: {"model": ScheduleOcrErrorResponse},
        503: {"model": ScheduleOcrErrorResponse},
        504: {"model": ScheduleOcrErrorResponse},
    },
)
async def recognize_schedule(
    multipart: ScheduleOcrMultipartRequest = Depends(parse_schedule_ocr_form),
    service: ScheduleOcrService = Depends(get_schedule_ocr_service),
) -> ScheduleOcrResponse:
    try:
        image = multipart.image
        fields = multipart.fields
        image_bytes = await image.read(service.max_image_bytes + 1)
        return service.recognize(
            image_bytes=image_bytes,
            content_type=image.content_type,
            filename=image.filename,
            year_month=fields.yearMonth,
            template_id=fields.templateId,
            row_index=fields.rowIndex,
            expected_width=fields.expectedWidth,
            expected_height=fields.expectedHeight,
            expected_sha256=fields.expectedSha256,
        )
    finally:
        await multipart.close()
