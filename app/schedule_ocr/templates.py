from dataclasses import dataclass

from app.schedule_ocr.errors import invalid_request


@dataclass(frozen=True, slots=True)
class ScheduleTemplate:
    template_id: str
    width: int
    height: int
    grid_left: int
    grid_top: int
    grid_right: int
    grid_bottom: int
    row_count: int
    column_count: int = 31


# Synthetic, non-identifying MVP template. Tune only from an approved production template.
FIXED_TEMPLATE_V1 = ScheduleTemplate(
    template_id="NURSE_HAND_FIXED_V1",
    width=1600,
    height=1200,
    grid_left=160,
    grid_top=300,
    grid_right=1536,
    grid_bottom=1040,
    row_count=16,
)

SUPPORTED_TEMPLATES = {FIXED_TEMPLATE_V1.template_id: FIXED_TEMPLATE_V1}


def get_template(template_id: str) -> ScheduleTemplate:
    template = SUPPORTED_TEMPLATES.get(template_id)
    if template is None:
        raise invalid_request("지원하지 않는 templateId입니다.")
    return template
