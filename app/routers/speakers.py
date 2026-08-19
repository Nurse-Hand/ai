from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.deps import get_diarization_service, get_embedding_service, get_speaker_store
from app.schemas import DeleteSpeakerResponse, RegisterDiarizedSpeakerResponse, SpeakersListResponse
from app.services.audio import copy_sample, normalize_audio, persist_upload
from app.services.diarization import DiarizationService
from app.services.speaker_embedding import SpeakerEmbeddingService
from app.services.speaker_store import SpeakerStore

router = APIRouter(prefix="/api/speakers", tags=["speakers"])


@router.get("", response_model=SpeakersListResponse)
def list_speakers(speaker_store: SpeakerStore = Depends(get_speaker_store)):
    speakers = speaker_store.list()
    return SpeakersListResponse(total=len(speakers), cacheMode="json_file", speakers=speakers)


@router.delete("/{speaker_id}", response_model=DeleteSpeakerResponse)
def delete_speaker(speaker_id: str, speaker_store: SpeakerStore = Depends(get_speaker_store)):
    deleted = speaker_store.delete(speaker_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="speaker not found")
    total = len(speaker_store.list())
    return DeleteSpeakerResponse(deletedSpeakerId=speaker_id, total=total, cacheMode="json_file")


@router.post("/register-from-diarization", response_model=RegisterDiarizedSpeakerResponse)
async def register_from_diarization(
    audio: UploadFile = File(...),
    diarizedSpeaker: str = Form(...),
    speakerId: str = Form(...),
    displayName: str = Form(...),
    settings: Settings = Depends(get_settings),
    diarization_service: DiarizationService = Depends(get_diarization_service),
    embedding_service: SpeakerEmbeddingService = Depends(get_embedding_service),
    speaker_store: SpeakerStore = Depends(get_speaker_store),
):
    job_dir = settings.tmp_dir / f"register-{uuid4()}"
    uploaded_path = await persist_upload(audio, job_dir)
    normalized_path = job_dir / "normalized-16k-mono.wav"
    normalize_audio(uploaded_path, normalized_path, settings)

    diarization_available, segments = diarization_service.analyze(normalized_path)
    selected_segments = [
        (segment.start_sec, segment.end_sec)
        for segment in segments
        if segment.diarized_speaker == diarizedSpeaker
    ]
    if not selected_segments:
        raise HTTPException(
            status_code=404,
            detail=f"diarizedSpeaker {diarizedSpeaker} not found in uploaded audio",
        )

    embedding = embedding_service.extract_embedding(normalized_path, selected_segments)
    sample_path = settings.data_dir / "speaker-samples" / f"{speakerId}.wav"
    copy_sample(normalized_path, sample_path)
    record = speaker_store.upsert(
        speaker_id=speakerId,
        display_name=displayName,
        embedding=embedding,
        sample_path=str(sample_path),
    )
    total_speech_sec = round(sum(end - start for start, end in selected_segments), 3)

    return RegisterDiarizedSpeakerResponse(
        speaker={
            "speakerId": record.speakerId,
            "displayName": record.displayName,
            "registeredAt": record.registeredAt,
            "updatedAt": record.updatedAt,
        },
        diarization={
            "diarizedSpeaker": diarizedSpeaker,
            "segmentCount": len(selected_segments),
            "totalSpeechSec": total_speech_sec,
            "rawSegmentCount": len(segments),
            "minSegmentSec": settings.min_segment_sec,
            "minSpeakerTotalSec": settings.min_speaker_total_sec,
            "maxSpeakerTotalSec": settings.max_speaker_total_sec,
            "diarizationAvailable": diarization_available,
        },
        embedding={"dimensions": len(embedding), "backend": embedding_service.backend_name},
        cache={"totalSpeakers": len(speaker_store.list()), "cacheMode": "json_file"},
    )
