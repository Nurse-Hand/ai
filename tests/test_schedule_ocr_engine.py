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


def synthetic_grid(image_format: str = "PNG", selected_row: int = 3) -> bytes:
    template = FIXED_TEMPLATE_V1
    image = Image.new("RGB", (template.width, template.height), "white")
    draw = ImageDraw.Draw(image)
    row_height = (template.grid_bottom - template.grid_top) / template.row_count
    for row in range(template.row_count):
        top = round(template.grid_top + row * row_height)
        bottom = round(template.grid_top + (row + 1) * row_height)
        if row != selected_row:
            draw.rectangle((template.grid_left, top, template.grid_right, bottom), fill="black")
    output = BytesIO()
    image.save(output, format=image_format, quality=95)
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
    assert bomb.value.status_code == 422


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
