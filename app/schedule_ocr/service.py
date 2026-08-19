import calendar
import re
from datetime import date

from PIL import Image, ImageOps

from app.schedule_ocr.engine import CellOcrEngine
from app.schedule_ocr.errors import invalid_request
from app.schedule_ocr.image import decode_image
from app.schedule_ocr.schemas import ScheduleOcrCell, ScheduleOcrResponse
from app.schedule_ocr.templates import ScheduleTemplate, get_template

YEAR_MONTH_PATTERN = re.compile(r"^(?P<year>20\d{2})-(?P<month>0[1-9]|1[0-2])$")


def selected_cells(image: Image.Image, template: ScheduleTemplate, row_index: int, day_count: int) -> list[Image.Image]:
    if row_index < 0 or row_index >= template.row_count:
        raise invalid_request("rowIndex가 template 범위를 벗어났습니다.")
    normalized = image.resize((template.width, template.height), Image.Resampling.LANCZOS)
    row_height = (template.grid_bottom - template.grid_top) / template.row_count
    row_top = round(template.grid_top + row_index * row_height)
    row_bottom = round(template.grid_top + (row_index + 1) * row_height)
    selected_row = normalized.crop((template.grid_left, row_top, template.grid_right, row_bottom))
    cell_width = selected_row.width / template.column_count

    cells: list[Image.Image] = []
    for column in range(day_count):
        left = round(column * cell_width)
        right = round((column + 1) * cell_width)
        cell = selected_row.crop((left, 0, right, selected_row.height))
        margin_x = max(1, round(cell.width * 0.08))
        margin_y = max(1, round(cell.height * 0.08))
        cell = cell.crop((margin_x, margin_y, cell.width - margin_x, cell.height - margin_y))
        gray = ImageOps.autocontrast(ImageOps.grayscale(cell))
        thresholded = gray.point(lambda value: 255 if value >= 180 else 0, mode="1")
        cells.append(thresholded.resize((thresholded.width * 3, thresholded.height * 3), Image.Resampling.NEAREST))
    return cells


class ScheduleOcrService:
    def __init__(
        self,
        engine: CellOcrEngine,
        *,
        max_image_bytes: int,
        min_image_width: int,
        min_image_height: int,
        max_image_pixels: int,
        review_threshold: float,
    ) -> None:
        self.engine = engine
        self.max_image_bytes = max_image_bytes
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height
        self.max_image_pixels = max_image_pixels
        self.review_threshold = review_threshold

    def recognize(
        self,
        *,
        image_bytes: bytes,
        content_type: str | None,
        filename: str | None,
        year_month: str,
        template_id: str,
        row_index: int,
    ) -> ScheduleOcrResponse:
        if not image_bytes:
            raise invalid_request("image가 비어 있습니다.")
        if len(image_bytes) > self.max_image_bytes:
            raise invalid_request("image 크기가 최대 기준을 초과합니다.")
        match = YEAR_MONTH_PATTERN.fullmatch(year_month)
        if match is None:
            raise invalid_request("yearMonth는 YYYY-MM 형식이어야 합니다.")
        year, month = int(match.group("year")), int(match.group("month"))
        template = get_template(template_id)
        image = decode_image(
            image_bytes,
            content_type=content_type,
            filename=filename,
            min_width=self.min_image_width,
            min_height=self.min_image_height,
            max_pixels=self.max_image_pixels,
        )
        cells = selected_cells(image, template, row_index, calendar.monthrange(year, month)[1])

        response_cells: list[ScheduleOcrCell] = []
        for day, cell in enumerate(cells, start=1):
            candidate = self.engine.recognize(cell)
            response_cells.append(
                ScheduleOcrCell(
                    date=date(year, month, day),
                    token=candidate.token,
                    confidence=candidate.confidence,
                    needsReview=candidate.token == "UNKNOWN" or candidate.confidence < self.review_threshold,
                )
            )

        warnings = ["SYNTHETIC_TEMPLATE_COORDINATES_REQUIRE_TUNING"]
        if any(cell.needsReview for cell in response_cells):
            warnings.append("REVIEW_REQUIRED")
        return ScheduleOcrResponse(
            templateId=template.template_id,
            yearMonth=year_month,
            cells=response_cells,
            warnings=warnings,
        )
