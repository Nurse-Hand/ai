from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "Nurse Hand AI Server"

    # 화자분리/음성분석 (backend-ai에서 이식, 2026-08-19)
    data_dir: Path = Path("./data")
    tmp_dir: Path = Path("./tmp")
    ffmpeg_bin: str = "ffmpeg"
    deepgram_api_key: Optional[str] = None
    deepgram_model: str = "nova-3"
    deepgram_language: str = "ko-KR"
    local_stt_model_dir: Optional[Path] = None
    pyannote_auth_token: Optional[str] = None
    pyannote_diarization_model: str = "pyannote/speaker-diarization-community-1"
    match_threshold: float = 0.7
    top_k_default: int = 3
    min_segment_sec: float = 0.8
    min_speaker_total_sec: float = 1.2
    max_speaker_total_sec: Optional[float] = None
    keep_uploads: bool = False

    # 업무/인수인계 AI 판단 로직
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    ai_timeout_seconds: float = 30.0
    internal_token: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
