from main import app
from app.schedule_ocr.errors import ScheduleOcrError


def test_main_app_registers_schedule_ocr_contract_and_error_handler() -> None:
    assert "/internal/v1/schedules/ocr" in app.openapi()["paths"]
    assert ScheduleOcrError in app.exception_handlers
