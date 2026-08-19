from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ScheduleOcrSettings(BaseSettings):
    tesseract_bin: str = "tesseract"
    tesseract_language: str = "eng"
    schedule_ocr_confidence_threshold: float = 0.85
    schedule_ocr_max_image_bytes: int = 10 * 1024 * 1024
    schedule_ocr_min_image_width: int = 640
    schedule_ocr_min_image_height: int = 480
    schedule_ocr_max_image_pixels: int = 16_000_000
    schedule_ocr_timeout_seconds: float = 5.0
    schedule_ocr_cell_timeout_seconds: float = 1.0
    schedule_ocr_max_concurrency: int = 2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_schedule_ocr_settings() -> ScheduleOcrSettings:
    return ScheduleOcrSettings()
