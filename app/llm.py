import os
from typing import TypeVar
from pydantic import BaseModel

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

    completion = client.beta.chat.completions.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format=response_model,
    )
    return completion.choices[0].message.parsed
