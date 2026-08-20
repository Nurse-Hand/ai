from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.config import Settings, get_settings
from app.deps import get_diarization_service, get_embedding_service, get_speaker_store, get_transcription_service
from app.schemas import RoundingDiarizationAnalysis
from app.services.analysis import build_analysis_response
from app.services.audio import normalize_audio, persist_upload
from app.services.diarization import DiarizationService
from app.services.speaker_embedding import SpeakerEmbeddingService
from app.services.speaker_store import SpeakerStore
from app.services.transcription import TranscriptionService

router = APIRouter(prefix="/api/diarization", tags=["diarization"])


@router.post("/analyze", response_model=RoundingDiarizationAnalysis)
async def analyze_diarization(
    audio: UploadFile = File(...),
    topK: int = Form(3),
    settings: Settings = Depends(get_settings),
    transcription_service: TranscriptionService = Depends(get_transcription_service),
    diarization_service: DiarizationService = Depends(get_diarization_service),
    embedding_service: SpeakerEmbeddingService = Depends(get_embedding_service),
    speaker_store: SpeakerStore = Depends(get_speaker_store),
):
    job_dir = settings.tmp_dir / f"analyze-{uuid4()}"
    uploaded_path = await persist_upload(audio, job_dir)
    normalized_path = job_dir / "normalized-16k-mono.wav"
    normalize_audio(uploaded_path, normalized_path, settings)

    transcript, utterances = await transcription_service.transcribe(normalized_path)
    diarization_available, raw_segments = diarization_service.analyze(normalized_path)
    registered_speakers = speaker_store.list()

    speaker_embeddings: Dict[str, Optional[List[float]]] = {}
    diarized_speakers = sorted({segment.diarized_speaker for segment in raw_segments})
    for diarized_speaker in diarized_speakers:
        selected_segments = [
            (segment.start_sec, segment.end_sec)
            for segment in raw_segments
            if segment.diarized_speaker == diarized_speaker
        ]
        try:
            speaker_embeddings[diarized_speaker] = embedding_service.extract_embedding(
                normalized_path,
                selected_segments,
            )
        except Exception:
            speaker_embeddings[diarized_speaker] = None

    return build_analysis_response(
        file_name=Path(audio.filename or "session.wav").name,
        diarization_available=diarization_available,
        transcript=transcript,
        utterances=utterances,
        raw_segments=raw_segments,
        speaker_embeddings=speaker_embeddings,
        registered_speakers=registered_speakers,
        embedding_service=embedding_service,
        settings=settings,
        top_k=max(1, topK or settings.top_k_default),
    )
