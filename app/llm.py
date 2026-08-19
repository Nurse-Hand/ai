from typing import TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.errors import InferenceFailure, InferenceFailureCode

T = TypeVar("T", bound=BaseModel)


def call_structured(
    system_prompt: str,
    user_content: str,
    response_model: type[T],
    _legacy_stub: T | None = None,
) -> T:
    """Call the configured model without logging prompts or source payloads."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise InferenceFailure(InferenceFailureCode.UNAVAILABLE)
    try:
        client = OpenAI(api_key=settings.openai_api_key, timeout=settings.ai_timeout_seconds)
        completion = client.beta.chat.completions.parse(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format=response_model,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise InferenceFailure(InferenceFailureCode.INVALID_RESPONSE)
        return parsed
    except InferenceFailure:
        raise
    except APITimeoutError as error:
        raise InferenceFailure(InferenceFailureCode.TIMEOUT) from error
    except RateLimitError as error:
        raise InferenceFailure(InferenceFailureCode.RATE_LIMITED) from error
    except APIConnectionError as error:
        raise InferenceFailure(InferenceFailureCode.UNAVAILABLE) from error
    except APIStatusError as error:
        code = InferenceFailureCode.UNAVAILABLE if error.status_code >= 500 else InferenceFailureCode.INVALID_RESPONSE
        raise InferenceFailure(code) from error
    except (LengthFinishReasonError, ContentFilterFinishReasonError) as error:
        raise InferenceFailure(InferenceFailureCode.INVALID_RESPONSE) from error
    except (ValidationError, ValueError, IndexError, AttributeError) as error:
        raise InferenceFailure(InferenceFailureCode.INVALID_RESPONSE) from error
