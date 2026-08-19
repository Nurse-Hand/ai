from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from app.auth import verify_internal_token
from app.schedule_ocr.engine import TesseractCellOcrEngine
from app.schedule_ocr.errors import ScheduleOcrError, invalid_request
from app.schedule_ocr.schemas import ScheduleOcrErrorResponse, ScheduleOcrResponse
from app.schedule_ocr.service import ScheduleOcrService
from app.schedule_ocr.settings import ScheduleOcrSettings, get_schedule_ocr_settings

router = APIRouter(
    prefix="/internal/v1/schedules",
    tags=["schedule-ocr"],
    dependencies=[Depends(verify_internal_token)],
)

ERROR_EXAMPLE = {"error": {"code": "SCHEDULE_OCR_INVALID_REQUEST", "message": "요청을 처리할 수 없습니다."}}
MULTIPART_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["image", "yearMonth", "templateId", "rowIndex"],
                    "properties": {
                        "image": {"type": "string", "format": "binary"},
                        "yearMonth": {"type": "string", "pattern": "^20[0-9]{2}-(0[1-9]|1[0-2])$", "example": "2026-08"},
                        "templateId": {"type": "string", "enum": ["NURSE_HAND_FIXED_V1"]},
                        "rowIndex": {"type": "integer", "minimum": 0, "maximum": 15, "example": 3},
                    },
                }
            }
        },
    }
}


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
        400: {"model": ScheduleOcrErrorResponse, "content": {"application/json": {"example": ERROR_EXAMPLE}}},
        415: {"model": ScheduleOcrErrorResponse},
        422: {"model": ScheduleOcrErrorResponse},
        502: {"model": ScheduleOcrErrorResponse},
        503: {"model": ScheduleOcrErrorResponse},
        504: {"model": ScheduleOcrErrorResponse},
    },
)
async def recognize_schedule(
    image: UploadFile | None = File(None),
    yearMonth: str | None = Form(None),
    templateId: str | None = Form(None),
    rowIndex: str | None = Form(None),
    service: ScheduleOcrService = Depends(get_schedule_ocr_service),
) -> ScheduleOcrResponse:
    if image is None or yearMonth is None or templateId is None or rowIndex is None:
        raise invalid_request("image, yearMonth, templateId, rowIndex는 필수입니다.")
    try:
        parsed_row_index = int(rowIndex)
    except ValueError as exc:
        raise invalid_request("rowIndex는 정수여야 합니다.") from exc

    image_bytes = await image.read(service.max_image_bytes + 1)
    return service.recognize(
        image_bytes=image_bytes,
        content_type=image.content_type,
        filename=image.filename,
        year_month=yearMonth,
        template_id=templateId,
        row_index=parsed_row_index,
    )
