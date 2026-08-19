import calendar
import re
from datetime import date

from PIL import Image, ImageOps

from app.schedule_ocr.engine import CellOcrEngine
from app.schedule_ocr.errors import invalid_request, unsupported_template
from app.schedule_ocr.image import decode_image
from app.schedule_ocr.schemas import ScheduleOcrCell, ScheduleOcrResponse
from app.schedule_ocr.templates import ScheduleTemplate, get_template

YEAR_MONTH_PATTERN = re.compile(r"^(?P<year>20\d{2})-(?P<month>0[1-9]|1[0-2])$")


def _dark_ratio(image: Image.Image) -> float:
    histogram = image.histogram()
    return histogram[1] / (image.width * image.height)


def _stripe_ratio(
    pixels,
    *,
    position: int,
    band: int,
    start: int,
    end: int,
    vertical: bool,
    width: int,
    height: int,
) -> float:
    if vertical:
        low, high = max(0, position - band), min(width, position + band + 1)
        dark = sum(pixels[x, y] for x in range(low, high) for y in range(start, end))
        return dark / ((high - low) * (end - start))
    low, high = max(0, position - band), min(height, position + band + 1)
    dark = sum(pixels[x, y] for y in range(low, high) for x in range(start, end))
    return dark / ((high - low) * (end - start))


def validate_template_structure(image: Image.Image, template: ScheduleTemplate) -> Image.Image:
    expected_ratio = template.width / template.height
    actual_ratio = image.width / image.height
    if abs(actual_ratio - expected_ratio) / expected_ratio > template.aspect_ratio_tolerance:
        raise unsupported_template("이미지 종횡비가 지원 template과 일치하지 않습니다.")

    normalized = image.resize((template.width, template.height), Image.Resampling.LANCZOS)
    grayscale = ImageOps.grayscale(normalized)
    dark = grayscale.point(lambda value: 1 if value < 96 else 0, mode="L")
    anchor_inner = dark.crop(template.anchor_inner_box)
    anchor_outer = dark.crop(template.anchor_outer_box)
    inner_dark_pixels = anchor_inner.histogram()[1]
    outer_dark_pixels = anchor_outer.histogram()[1]
    surround_pixels = anchor_outer.width * anchor_outer.height - anchor_inner.width * anchor_inner.height
    surround_dark_ratio = (outer_dark_pixels - inner_dark_pixels) / surround_pixels
    if (
        _dark_ratio(anchor_inner) < template.minimum_anchor_inner_dark_ratio
        or surround_dark_ratio > template.maximum_anchor_surround_dark_ratio
    ):
        raise unsupported_template("지원 template anchor를 찾을 수 없습니다.")

    pixels = dark.load()
    band = 2
    vertical_positions = [
        round(template.grid_left + column * (template.grid_right - template.grid_left) / template.column_count)
        for column in range(template.column_count + 1)
    ]
    horizontal_positions = [
        round(template.grid_top + row * (template.grid_bottom - template.grid_top) / template.row_count)
        for row in range(template.row_count + 1)
    ]
    adjacent_offset = 8
    vertical_ratios = []
    vertical_contrasts = []
    for index, position in enumerate(vertical_positions):
        line_ratio = _stripe_ratio(
            pixels, position=position, band=band, start=template.grid_top,
            end=template.grid_bottom, vertical=True, width=template.width, height=template.height,
        )
        adjacent_position = position + adjacent_offset if index < len(vertical_positions) - 1 else position - adjacent_offset
        adjacent_ratio = _stripe_ratio(
            pixels, position=adjacent_position, band=band, start=template.grid_top,
            end=template.grid_bottom, vertical=True, width=template.width, height=template.height,
        )
        vertical_ratios.append(line_ratio)
        vertical_contrasts.append(line_ratio - adjacent_ratio)

    horizontal_ratios = []
    horizontal_contrasts = []
    for index, position in enumerate(horizontal_positions):
        line_ratio = _stripe_ratio(
            pixels, position=position, band=band, start=template.grid_left,
            end=template.grid_right, vertical=False, width=template.width, height=template.height,
        )
        adjacent_position = position + adjacent_offset if index < len(horizontal_positions) - 1 else position - adjacent_offset
        adjacent_ratio = _stripe_ratio(
            pixels, position=adjacent_position, band=band, start=template.grid_left,
            end=template.grid_right, vertical=False, width=template.width, height=template.height,
        )
        horizontal_ratios.append(line_ratio)
        horizontal_contrasts.append(line_ratio - adjacent_ratio)

    if (
        min(vertical_ratios) < template.minimum_grid_line_ratio
        or min(horizontal_ratios) < template.minimum_grid_line_ratio
        or min(vertical_contrasts) < template.minimum_grid_contrast
        or min(horizontal_contrasts) < template.minimum_grid_contrast
    ):
        raise unsupported_template("지원 template 격자 구조를 찾을 수 없습니다.")

    cell_width = (template.grid_right - template.grid_left) / template.column_count
    cell_height = (template.grid_bottom - template.grid_top) / template.row_count
    for row in range(template.row_count):
        for column in range(template.column_count):
            left = round(template.grid_left + column * cell_width + band + 4)
            right = round(template.grid_left + (column + 1) * cell_width - band - 4)
            top = round(template.grid_top + row * cell_height + band + 4)
            bottom = round(template.grid_top + (row + 1) * cell_height - band - 4)
            cell = grayscale.crop((left, top, right, bottom))
            histogram = cell.histogram()
            bright_ratio = sum(histogram[160:]) / (cell.width * cell.height)
            if bright_ratio < template.minimum_cell_bright_ratio:
                raise unsupported_template("지원 template 셀 내부 밝기 구조가 일치하지 않습니다.")
    return normalized


def selected_cells(normalized: Image.Image, template: ScheduleTemplate, row_index: int, day_count: int) -> list[Image.Image]:
    if row_index < 0 or row_index >= template.row_count:
        raise invalid_request("rowIndex가 template 범위를 벗어났습니다.")
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
        normalized = validate_template_structure(image, template)
        cells = selected_cells(normalized, template, row_index, calendar.monthrange(year, month)[1])

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
