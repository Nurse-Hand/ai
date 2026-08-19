from io import BytesIO
import shutil
import subprocess

from PIL import Image, ImageDraw
import pytest

from app.schedule_ocr.engine import OcrCandidate, TesseractCellOcrEngine, normalize_token
from app.schedule_ocr.errors import ScheduleOcrError
from app.schedule_ocr.service import ScheduleOcrService
from app.schedule_ocr.templates import FIXED_TEMPLATE_V1


class RecordingEngine:
    def __init__(self, candidate: OcrCandidate = OcrCandidate("D", 0.95)) -> None:
        self.candidate = candidate
        self.cells: list[Image.Image] = []

    def recognize(self, cell: Image.Image) -> OcrCandidate:
        self.cells.append(cell.copy())
        return self.candidate


def synthetic_grid(
    image_format: str = "PNG",
    selected_row: int = 3,
    *,
    grid_offset: int = 0,
    include_anchor: bool = True,
    exif_orientation: int | None = None,
) -> bytes:
    template = FIXED_TEMPLATE_V1
    image = Image.new("RGB", (template.width, template.height), "white")
    draw = ImageDraw.Draw(image)
    row_height = (template.grid_bottom - template.grid_top) / template.row_count
    if include_anchor:
        draw.rectangle((60, 56, 132, 104), fill="black")
    for row in range(template.row_count):
        top = round(template.grid_top + row * row_height)
        bottom = round(template.grid_top + (row + 1) * row_height)
        if row != selected_row:
            draw.rectangle((template.grid_left + 12, top + 12, template.grid_left + 24, bottom - 12), fill="black")
    for column in range(template.column_count + 1):
        x = round(template.grid_left + column * (template.grid_right - template.grid_left) / template.column_count) + grid_offset
        draw.line((x, template.grid_top + grid_offset, x, template.grid_bottom + grid_offset), fill="black", width=5)
    for row in range(template.row_count + 1):
        y = round(template.grid_top + row * row_height) + grid_offset
        draw.line((template.grid_left + grid_offset, y, template.grid_right + grid_offset, y), fill="black", width=5)
    if exif_orientation is not None:
        rotation = {3: 180, 6: 90, 8: 270}[exif_orientation]
        image = image.rotate(rotation, expand=True)
        exif = image.getexif()
        exif[274] = exif_orientation
    else:
        exif = None
    output = BytesIO()
    save_options = {"quality": 95}
    if exif is not None:
        save_options["exif"] = exif
    image.save(output, format=image_format, **save_options)
    return output.getvalue()


def service(engine: RecordingEngine, **overrides: int | float) -> ScheduleOcrService:
    options: dict[str, int | float] = {
        "max_image_bytes": 10 * 1024 * 1024,
        "min_image_width": 640,
        "min_image_height": 480,
        "max_image_pixels": 16_000_000,
        "review_threshold": 0.85,
    }
    options.update(overrides)
    return ScheduleOcrService(engine, **options)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("year_month", "expected_days"),
    [("2026-02", 28), ("2024-02", 29), ("2026-04", 30), ("2026-01", 31)],
)
def test_month_day_counts_and_selected_row_only(year_month: str, expected_days: int) -> None:
    engine = RecordingEngine()
    result = service(engine).recognize(
        image_bytes=synthetic_grid(selected_row=3),
        content_type="image/png",
        filename="synthetic.png",
        year_month=year_month,
        template_id="NURSE_HAND_FIXED_V1",
        row_index=3,
    )
    assert len(result.cells) == expected_days
    assert len(engine.cells) == expected_days
    assert all(cell.getextrema() == (255, 255) for cell in engine.cells)


@pytest.mark.parametrize(("raw", "confidence", "token"), [
    ("D", 0.9, "D"), ("e", 0.9, "E"), ("N", 0.9, "N"), ("O F F", 0.9, "OFF"),
    ("O", 0.99, "UNKNOWN"), ("0", 0.99, "UNKNOWN"), ("DAY", 0.99, "UNKNOWN"),
    ("D", 0.84, "UNKNOWN"), ("", 0.99, "UNKNOWN"),
])
def test_token_allowlist_and_confidence(raw: str, confidence: float, token: str) -> None:
    assert normalize_token(raw, confidence, 0.85).token == token


def test_unknown_always_needs_review() -> None:
    result = service(RecordingEngine(OcrCandidate("UNKNOWN", 0.99))).recognize(
        image_bytes=synthetic_grid(), content_type="image/png", filename="synthetic.png",
        year_month="2026-02", template_id="NURSE_HAND_FIXED_V1", row_index=3,
    )
    assert all(cell.needsReview for cell in result.cells)
    assert "REVIEW_REQUIRED" in result.warnings


@pytest.mark.parametrize(
    ("internal_token", "wire_token"),
    [("D", "DAY"), ("E", "EVENING"), ("N", "NIGHT"), ("OFF", "OFF"), ("UNKNOWN", "UNKNOWN")],
)
def test_internal_candidates_map_to_wire_tokens(internal_token: str, wire_token: str) -> None:
    candidate = OcrCandidate(internal_token, 0.95)  # type: ignore[arg-type]
    result = service(RecordingEngine(candidate)).recognize(
        image_bytes=synthetic_grid(), content_type="image/png", filename="synthetic.png",
        year_month="2026-02", template_id="NURSE_HAND_FIXED_V1", row_index=3,
    )
    assert result.cells[0].token == wire_token


@pytest.mark.parametrize(("template_id", "row_index", "year_month"), [
    ("UNKNOWN", 0, "2026-01"), ("NURSE_HAND_FIXED_V1", -1, "2026-01"),
    ("NURSE_HAND_FIXED_V1", 16, "2026-01"), ("NURSE_HAND_FIXED_V1", 0, "2026-13"),
])
def test_template_row_and_month_bounds(template_id: str, row_index: int, year_month: str) -> None:
    with pytest.raises(ScheduleOcrError) as raised:
        service(RecordingEngine()).recognize(
            image_bytes=synthetic_grid(), content_type="image/png", filename="synthetic.png",
            year_month=year_month, template_id=template_id, row_index=row_index,
        )
    assert raised.value.status_code == 400


@pytest.mark.parametrize(("image_format", "content_type", "filename"), [
    ("PNG", "image/png", "synthetic.png"), ("JPEG", "image/jpeg", "synthetic.jpg"),
])
def test_png_and_jpeg_decode(image_format: str, content_type: str, filename: str) -> None:
    result = service(RecordingEngine()).recognize(
        image_bytes=synthetic_grid(image_format), content_type=content_type, filename=filename,
        year_month="2026-02", template_id="NURSE_HAND_FIXED_V1", row_index=3,
    )
    assert len(result.cells) == 28


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_exif_orientation_is_normalized_before_template_validation(orientation: int) -> None:
    result = service(RecordingEngine()).recognize(
        image_bytes=synthetic_grid("JPEG", exif_orientation=orientation),
        content_type="image/jpeg", filename="synthetic.jpg", year_month="2026-02",
        template_id="NURSE_HAND_FIXED_V1", row_index=3,
    )
    assert len(result.cells) == 28


@pytest.mark.parametrize("case", ["irrelevant", "aspect", "shifted-grid", "missing-anchor"])
def test_unrelated_or_misaligned_images_are_unsupported_templates(case: str) -> None:
    if case == "irrelevant":
        image = Image.new("RGB", (1600, 1200), "white")
        output = BytesIO()
        image.save(output, "PNG")
        data = output.getvalue()
    elif case == "aspect":
        image = Image.new("RGB", (1200, 1200), "white")
        output = BytesIO()
        image.save(output, "PNG")
        data = output.getvalue()
    elif case == "shifted-grid":
        data = synthetic_grid(grid_offset=12)
    else:
        data = synthetic_grid(include_anchor=False)
    with pytest.raises(ScheduleOcrError) as raised:
        service(RecordingEngine()).recognize(
            image_bytes=data, content_type="image/png", filename="synthetic.png",
            year_month="2026-02", template_id="NURSE_HAND_FIXED_V1", row_index=3,
        )
    assert raised.value.code == "SCHEDULE_OCR_UNSUPPORTED_TEMPLATE"


@pytest.mark.parametrize("case", ["black", "black-grid", "dark-unrelated"])
def test_uniform_dark_images_cannot_impersonate_template(case: str) -> None:
    template = FIXED_TEMPLATE_V1
    if case == "black":
        image = Image.new("RGB", (template.width, template.height), "black")
    elif case == "dark-unrelated":
        image = Image.new("RGB", (template.width, template.height), (48, 48, 48))
    else:
        image = Image.open(BytesIO(synthetic_grid())).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (template.grid_left, template.grid_top, template.grid_right, template.grid_bottom),
            fill="black",
        )
    output = BytesIO()
    image.save(output, "PNG")
    with pytest.raises(ScheduleOcrError) as raised:
        service(RecordingEngine()).recognize(
            image_bytes=output.getvalue(), content_type="image/png", filename="synthetic.png",
            year_month="2026-02", template_id="NURSE_HAND_FIXED_V1", row_index=3,
        )
    assert raised.value.code == "SCHEDULE_OCR_UNSUPPORTED_TEMPLATE"


def test_jpeg_fill_byte_before_app_marker_is_accepted() -> None:
    jpeg = synthetic_grid("JPEG")
    with_fill_byte = jpeg[:2] + b"\xff" + jpeg[2:]
    result = service(RecordingEngine()).recognize(
        image_bytes=with_fill_byte, content_type="image/jpeg", filename="synthetic.jpg",
        year_month="2026-02", template_id="NURSE_HAND_FIXED_V1", row_index=3,
    )
    assert len(result.cells) == 28


def test_signature_spoof_is_rejected() -> None:
    with pytest.raises(ScheduleOcrError) as raised:
        service(RecordingEngine()).recognize(
            image_bytes=synthetic_grid("PNG"), content_type="image/jpeg", filename="synthetic.jpg",
            year_month="2026-02", template_id="NURSE_HAND_FIXED_V1", row_index=3,
        )
    assert raised.value.status_code == 415


def test_truncated_image_is_rejected() -> None:
    with pytest.raises(ScheduleOcrError) as raised:
        service(RecordingEngine()).recognize(
            image_bytes=synthetic_grid()[:100], content_type="image/png", filename="synthetic.png",
            year_month="2026-02", template_id="NURSE_HAND_FIXED_V1", row_index=3,
        )
    assert raised.value.status_code == 422


def test_oversize_and_pixel_bomb_are_rejected() -> None:
    data = synthetic_grid()
    with pytest.raises(ScheduleOcrError) as oversize:
        service(RecordingEngine(), max_image_bytes=10).recognize(
            image_bytes=data, content_type="image/png", filename="synthetic.png",
            year_month="2026-02", template_id="NURSE_HAND_FIXED_V1", row_index=3,
        )
    assert oversize.value.status_code == 400
    with pytest.raises(ScheduleOcrError) as bomb:
        service(RecordingEngine(), max_image_pixels=1_000_000).recognize(
            image_bytes=data, content_type="image/png", filename="synthetic.png",
            year_month="2026-02", template_id="NURSE_HAND_FIXED_V1", row_index=3,
        )
    assert bomb.value.status_code == 415


def test_decode_does_not_mutate_global_pillow_pixel_limit() -> None:
    original = Image.MAX_IMAGE_PIXELS
    service(RecordingEngine()).recognize(
        image_bytes=synthetic_grid(), content_type="image/png", filename="synthetic.png",
        year_month="2026-02", template_id="NURSE_HAND_FIXED_V1", row_index=3,
    )
    assert Image.MAX_IMAGE_PIXELS == original


def tesseract_engine() -> TesseractCellOcrEngine:
    return TesseractCellOcrEngine("tesseract", "eng", 0.85, 1.0)


def test_cli_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _binary: None)
    with pytest.raises(ScheduleOcrError) as raised:
        tesseract_engine().recognize(Image.new("1", (20, 20), 1))
    assert raised.value.status_code == 503


def test_cli_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _binary: "tesseract")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("t", 1)))
    with pytest.raises(ScheduleOcrError) as raised:
        tesseract_engine().recognize(Image.new("1", (20, 20), 1))
    assert raised.value.status_code == 504


@pytest.mark.parametrize(("returncode", "stdout"), [(1, b""), (0, b"invalid-tsv")])
def test_cli_nonzero_and_invalid_tsv(monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: bytes) -> None:
    monkeypatch.setattr(shutil, "which", lambda _binary: "tesseract")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode, stdout))
    with pytest.raises(ScheduleOcrError) as raised:
        tesseract_engine().recognize(Image.new("1", (20, 20), 1))
    assert raised.value.status_code == 502


def test_cli_uses_memory_stdin_without_temp_files(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(shutil, "which", lambda _binary: "tesseract")

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        tsv = b"level\tconf\ttext\n5\t96\tOFF\n"
        return subprocess.CompletedProcess(command, 0, tsv)

    monkeypatch.setattr(subprocess, "run", fake_run)
    candidate = tesseract_engine().recognize(Image.new("1", (20, 20), 1))
    assert candidate == OcrCandidate("OFF", 0.96)
    assert captured["shell"] is False
    assert isinstance(captured["input"], bytes)
