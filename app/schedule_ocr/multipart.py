import re
from dataclasses import dataclass
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import Request
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError
from starlette.datastructures import UploadFile

from app.schedule_ocr.errors import invalid_request

FORM_FIELDS = frozenset(
    {
        "image",
        "yearMonth",
        "templateId",
        "rowIndex",
        "expectedWidth",
        "expectedHeight",
        "expectedSha256",
    }
)
CANONICAL_INTEGER_PATTERN = re.compile(r"0|[1-9][0-9]*")


def _canonical_integer(value: Any) -> int:
    if isinstance(value, str) and CANONICAL_INTEGER_PATTERN.fullmatch(value):
        return int(value)
    raise ValueError("canonical decimal integer required")


CanonicalInteger = Annotated[int, BeforeValidator(_canonical_integer)]


class ScheduleOcrFormFields(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    yearMonth: str = Field(pattern=r"^20[0-9]{2}-(0[1-9]|1[0-2])$")
    templateId: Literal["NURSE_HAND_FIXED_V1"]
    rowIndex: Annotated[CanonicalInteger, Field(ge=0, le=15)]
    expectedWidth: Annotated[CanonicalInteger, Field(gt=0)]
    expectedHeight: Annotated[CanonicalInteger, Field(gt=0)]
    expectedSha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ScheduleOcrMultipartRequest:
    image: UploadFile
    fields: ScheduleOcrFormFields


async def parse_schedule_ocr_form(request: Request) -> AsyncIterator[ScheduleOcrMultipartRequest]:
    form = await request.form()
    try:
        items = form.multi_items()
        keys = [key for key, _value in items]
        if set(keys) != FORM_FIELDS or len(keys) != len(FORM_FIELDS):
            raise invalid_request("multipart 필드가 계약과 일치하지 않습니다.")

        values = dict(items)
        image = values.pop("image")
        if not isinstance(image, UploadFile):
            raise invalid_request("image는 파일이어야 합니다.")
        if any(not isinstance(value, str) for value in values.values()):
            raise invalid_request("multipart 필드 형식이 올바르지 않습니다.")
        try:
            fields = ScheduleOcrFormFields.model_validate(values)
        except ValidationError as exc:
            raise invalid_request("multipart 필드 형식이 올바르지 않습니다.") from exc
        yield ScheduleOcrMultipartRequest(image=image, fields=fields)
    finally:
        await form.close()
