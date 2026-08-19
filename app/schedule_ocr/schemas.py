from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

InternalScheduleToken = Literal["D", "E", "N", "OFF", "UNKNOWN"]
ScheduleToken = Literal["DAY", "EVENING", "NIGHT", "OFF", "UNKNOWN"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ScheduleOcrCell(StrictModel):
    date: date
    token: ScheduleToken
    confidence: float = Field(ge=0.0, le=1.0)
    needsReview: bool


class ScheduleOcrResponse(StrictModel):
    contractVersion: Literal["schedule-ocr.v1"] = "schedule-ocr.v1"
    templateId: str
    yearMonth: str
    cells: list[ScheduleOcrCell]
    warnings: list[str]


class ErrorDetail(StrictModel):
    code: str
    message: str


class ScheduleOcrErrorResponse(StrictModel):
    error: ErrorDetail
