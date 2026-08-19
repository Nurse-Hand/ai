from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints
from pydantic.alias_generators import to_camel


def _require_trimmed_text(value: str) -> str:
    if value != value.strip():
        raise ValueError("text must not have leading or trailing whitespace")
    return value


def bounded_text(max_length: int, *, min_length: int = 1):
    return Annotated[
        str,
        StringConstraints(
            min_length=min_length,
            max_length=max_length,
            pattern=r"^[^\x00-\x1F\x7F]*$",
        ),
        AfterValidator(_require_trimmed_text),
    ]


class StrictCamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )
