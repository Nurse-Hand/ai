from enum import Enum

from pydantic import BaseModel, ConfigDict


class InferenceFailureCode(str, Enum):
    TIMEOUT = "AI_UPSTREAM_TIMEOUT"
    RATE_LIMITED = "AI_UPSTREAM_RATE_LIMITED"
    UNAVAILABLE = "AI_UPSTREAM_UNAVAILABLE"
    INVALID_RESPONSE = "AI_UPSTREAM_INVALID_RESPONSE"


class InferenceFailure(RuntimeError):
    def __init__(self, code: InferenceFailureCode):
        super().__init__(code.value)
        self.code = code


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail


INTERNAL_ERROR_RESPONSES = {
    status: {"model": ErrorResponse}
    for status in (401, 422, 429, 502, 503, 504)
}
