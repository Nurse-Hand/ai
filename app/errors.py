from enum import Enum


class InferenceFailureCode(str, Enum):
    TIMEOUT = "AI_UPSTREAM_TIMEOUT"
    RATE_LIMITED = "AI_UPSTREAM_RATE_LIMITED"
    UNAVAILABLE = "AI_UPSTREAM_UNAVAILABLE"
    INVALID_RESPONSE = "AI_UPSTREAM_INVALID_RESPONSE"


class InferenceFailure(RuntimeError):
    def __init__(self, code: InferenceFailureCode):
        super().__init__(code.value)
        self.code = code
