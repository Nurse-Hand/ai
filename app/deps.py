from functools import lru_cache

from app.config import get_settings
from app.services.diarization import DiarizationService
from app.services.speaker_embedding import SpeakerEmbeddingService
from app.services.speaker_store import SpeakerStore
from app.services.transcription import TranscriptionService


@lru_cache(maxsize=1)
def get_diarization_service() -> DiarizationService:
    return DiarizationService(get_settings())


@lru_cache(maxsize=1)
def get_transcription_service() -> TranscriptionService:
    return TranscriptionService(get_settings())


@lru_cache(maxsize=1)
def get_embedding_service() -> SpeakerEmbeddingService:
    return SpeakerEmbeddingService()


@lru_cache(maxsize=1)
def get_speaker_store() -> SpeakerStore:
    return SpeakerStore(get_settings().data_dir)
