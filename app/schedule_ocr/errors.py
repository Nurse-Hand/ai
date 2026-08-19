from dataclasses import dataclass


@dataclass(slots=True)
class ScheduleOcrError(Exception):
    code: str
    message: str
    status_code: int


def invalid_request(message: str) -> ScheduleOcrError:
    return ScheduleOcrError("SCHEDULE_OCR_INVALID_REQUEST", message, 400)


def unsupported_image(message: str) -> ScheduleOcrError:
    return ScheduleOcrError("SCHEDULE_OCR_UNSUPPORTED_IMAGE", message, 415)


def decode_failed() -> ScheduleOcrError:
    return ScheduleOcrError("SCHEDULE_OCR_DECODE_FAILED", "이미지를 안전하게 해석할 수 없습니다.", 422)


def engine_unavailable() -> ScheduleOcrError:
    return ScheduleOcrError("SCHEDULE_OCR_ENGINE_UNAVAILABLE", "OCR 엔진을 사용할 수 없습니다.", 503)


def engine_timeout() -> ScheduleOcrError:
    return ScheduleOcrError("SCHEDULE_OCR_ENGINE_TIMEOUT", "OCR 엔진 처리 시간이 초과되었습니다.", 504)


def engine_failed() -> ScheduleOcrError:
    return ScheduleOcrError("SCHEDULE_OCR_ENGINE_FAILED", "OCR 엔진이 유효한 결과를 반환하지 않았습니다.", 502)
