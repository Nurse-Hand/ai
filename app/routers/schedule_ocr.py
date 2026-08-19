import asyncio
from concurrent.futures import Future
from functools import lru_cache
import logging
import threading
import time
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse

from app.auth import InternalAuthError, require_internal_token
from app.config import Settings, get_settings
from app.schedule_ocr.engine import TesseractCellOcrEngine
from app.schedule_ocr.errors import ScheduleOcrError, engine_busy, engine_timeout, engine_unavailable
from app.schedule_ocr.multipart import ScheduleOcrMultipartRequest, parse_schedule_ocr_form
from app.schedule_ocr.schemas import ScheduleOcrErrorResponse, ScheduleOcrResponse
from app.schedule_ocr.service import ScheduleOcrService
from app.schedule_ocr.settings import ScheduleOcrSettings, get_schedule_ocr_settings

ERROR_EXAMPLE = {"error": {"code": "SCHEDULE_OCR_INVALID_REQUEST", "message": "요청을 처리할 수 없습니다."}}
logger = logging.getLogger(__name__)
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
    settings: Settings = Depends(get_settings),
) -> None:
    try:
        require_internal_token(x_internal_token, settings.internal_token)
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


class ScheduleOcrInferenceGate:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("OCR inference capacity must be positive")
        self._semaphore = threading.BoundedSemaphore(capacity)
        self._workers: set[Future] = set()
        self._workers_lock = threading.Lock()

    @property
    def worker_count(self) -> int:
        with self._workers_lock:
            return len(self._workers)

    def try_accept(self) -> Future | None:
        if not self._semaphore.acquire(blocking=False):
            return None
        worker: Future = Future()
        with self._workers_lock:
            self._workers.add(worker)
        worker.add_done_callback(self._complete)
        return worker

    def _complete(self, worker: Future) -> None:
        try:
            worker.result()
        except BaseException:
            logger.warning("Internal schedule OCR worker failed.")
        finally:
            with self._workers_lock:
                self._workers.discard(worker)
            self._semaphore.release()

    def start(self, worker: Future, target, **kwargs) -> None:
        def run() -> None:
            if not worker.set_running_or_notify_cancel():
                return
            try:
                result = target(**kwargs)
            except BaseException as error:
                worker.set_exception(error)
            else:
                worker.set_result(result)

        thread = threading.Thread(target=run, name="internal-schedule-ocr", daemon=True)
        try:
            thread.start()
        except Exception as error:
            worker.set_exception(engine_unavailable())
            raise engine_unavailable() from error

    async def drain(self, timeout_seconds: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout_seconds)
        while self.worker_count:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning("Internal schedule OCR workers are still pending at shutdown.")
                return
            await asyncio.sleep(min(0.05, remaining))


_registered_inference_gates: set[ScheduleOcrInferenceGate] = set()
_registered_inference_gates_lock = threading.Lock()


@lru_cache(maxsize=None)
def _inference_gate(capacity: int) -> ScheduleOcrInferenceGate:
    gate = ScheduleOcrInferenceGate(capacity)
    with _registered_inference_gates_lock:
        _registered_inference_gates.add(gate)
    return gate


def get_schedule_ocr_inference_gate(
    settings: ScheduleOcrSettings = Depends(get_schedule_ocr_settings),
) -> ScheduleOcrInferenceGate:
    return _inference_gate(settings.schedule_ocr_max_concurrency)


async def drain_schedule_ocr_workers(timeout_seconds: float = 5.0) -> None:
    gates = list(_registered_inference_gates)
    if gates:
        await asyncio.gather(*(gate.drain(timeout_seconds) for gate in gates))


def get_schedule_ocr_service(
    settings: ScheduleOcrSettings = Depends(get_schedule_ocr_settings),
) -> ScheduleOcrService:
    return ScheduleOcrService(
        TesseractCellOcrEngine(
            settings.tesseract_bin,
            settings.tesseract_language,
            settings.schedule_ocr_confidence_threshold,
            settings.schedule_ocr_cell_timeout_seconds,
        ),
        max_image_bytes=settings.schedule_ocr_max_image_bytes,
        min_image_width=settings.schedule_ocr_min_image_width,
        min_image_height=settings.schedule_ocr_min_image_height,
        max_image_pixels=settings.schedule_ocr_max_image_pixels,
        review_threshold=settings.schedule_ocr_confidence_threshold,
        inference_timeout_seconds=settings.schedule_ocr_timeout_seconds,
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
    inference_gate: ScheduleOcrInferenceGate = Depends(get_schedule_ocr_inference_gate),
) -> ScheduleOcrResponse:
    image = multipart.image
    fields = multipart.fields
    image_bytes = await image.read(service.max_image_bytes + 1)
    worker = inference_gate.try_accept()
    if worker is None:
        raise engine_busy()
    deadline = time.monotonic() + service.inference_timeout_seconds
    inference_gate.start(
        worker,
        service.recognize,
        image_bytes=image_bytes,
        content_type=image.content_type,
        filename=image.filename,
        year_month=fields.yearMonth,
        template_id=fields.templateId,
        row_index=fields.rowIndex,
        expected_width=fields.expectedWidth,
        expected_height=fields.expectedHeight,
        expected_sha256=fields.expectedSha256,
        deadline=deadline,
    )
    wrapped_worker = asyncio.wrap_future(worker)
    try:
        return await asyncio.wait_for(
            asyncio.shield(wrapped_worker),
            timeout=max(0.0, deadline - time.monotonic()),
        )
    except asyncio.TimeoutError as error:
        raise engine_timeout() from error
