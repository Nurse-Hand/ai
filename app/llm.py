import os
import time
from typing import TypeVar
from pydantic import BaseModel
from openai import RateLimitError

T = TypeVar("T", bound=BaseModel)

_client = None


def _get_client():
    """OPENAI_API_KEY 없으면 None -> 호출부에서 스텁으로 대체."""
    global _client
    if _client is None and os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        _client = OpenAI()
    return _client


def call_structured(system_prompt: str, user_content: str, response_model: type[T], stub: T) -> T:
    """GPT-4o structured output 호출. 키 없으면 stub 그대로 반환."""
    client = _get_client()
    if client is None:
        return stub

    # ponytail: OpenAI 조직 tier 업그레이드 직후 일부 요청이 랜덤하게 예전 rate limit에
    # 걸리는 전파 지연 케이스가 관측됨 - 짧은 재시도로 흡수. 계속 실패하면 그대로 raise.
    last_error: RateLimitError | None = None
    for attempt in range(3):
        try:
            completion = client.beta.chat.completions.parse(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format=response_model,
                temperature=0,  # 분류/판단 작업이라 매번 같은 입력엔 같은 출력이 나와야 함 (재현성)
            )
            return completion.choices[0].message.parsed
        except RateLimitError as e:
            last_error = e
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise last_error
