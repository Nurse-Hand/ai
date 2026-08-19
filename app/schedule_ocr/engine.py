import csv
import io
import shutil
import subprocess
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from app.schedule_ocr.errors import engine_failed, engine_timeout, engine_unavailable
from app.schedule_ocr.schemas import ScheduleToken


@dataclass(frozen=True, slots=True)
class OcrCandidate:
    token: ScheduleToken
    confidence: float


class CellOcrEngine(Protocol):
    def recognize(self, cell: Image.Image) -> OcrCandidate: ...


def normalize_token(raw: str, confidence: float, threshold: float) -> OcrCandidate:
    compact = "".join(raw.upper().split())
    bounded = max(0.0, min(1.0, confidence))
    if compact in {"D", "E", "N", "OFF"} and bounded >= threshold:
        return OcrCandidate(compact, bounded)  # type: ignore[arg-type]
    return OcrCandidate("UNKNOWN", bounded)


class TesseractCellOcrEngine:
    def __init__(self, binary: str, language: str, confidence_threshold: float, timeout_seconds: float) -> None:
        self.binary = binary
        self.language = language
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds

    def recognize(self, cell: Image.Image) -> OcrCandidate:
        executable = shutil.which(self.binary)
        if executable is None:
            raise engine_unavailable()

        payload = io.BytesIO()
        cell.save(payload, format="PNG")
        command = [
            executable,
            "stdin",
            "stdout",
            "-l",
            self.language,
            "--psm",
            "10",
            "-c",
            "tessedit_char_whitelist=DENOF",
            "tsv",
        ]
        try:
            completed = subprocess.run(
                command,
                input=payload.getvalue(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise engine_timeout() from exc
        except OSError as exc:
            raise engine_unavailable() from exc
        if completed.returncode != 0:
            raise engine_failed()

        try:
            reader = csv.DictReader(io.StringIO(completed.stdout.decode("utf-8", errors="strict")), delimiter="\t")
            required = {"level", "conf", "text"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise engine_failed()
            words: list[tuple[str, float]] = []
            for row in reader:
                text = (row.get("text") or "").strip()
                confidence = float(row.get("conf") or -1)
                if text and confidence >= 0:
                    words.append((text, confidence / 100.0))
        except (UnicodeDecodeError, ValueError, TypeError, csv.Error) as exc:
            raise engine_failed() from exc

        if not words:
            return OcrCandidate("UNKNOWN", 0.0)
        return normalize_token(
            "".join(text for text, _ in words),
            min(score for _, score in words),
            self.confidence_threshold,
        )
