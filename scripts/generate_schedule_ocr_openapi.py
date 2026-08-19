import json
from pathlib import Path

from fastapi import FastAPI

from app.routers.schedule_ocr import router, schedule_ocr_error_handler
from app.schedule_ocr.errors import ScheduleOcrError

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "openapi" / "schedule-ocr.v1.json"


def contract_schema() -> dict:
    app = FastAPI(title="Nurse Hand Schedule OCR Contract", version="schedule-ocr.v1")
    app.include_router(router)
    app.add_exception_handler(ScheduleOcrError, schedule_ocr_error_handler)
    return app.openapi()


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(contract_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
