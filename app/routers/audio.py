import shutil
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.audio_contracts import (
    AnalyzeAudioResponse,
    AudioDiarizedSpeaker,
    AudioTranscript,
    AudioUtterance,
)
from app.auth import verify_internal_token
from app.config import Settings, get_settings
from app.deps import get_diarization_service, get_transcription_service
from app.errors import INTERNAL_ERROR_RESPONSES, InferenceFailure, InferenceFailureCode
from app.services.analysis import find_best_overlap
from app.services.audio import normalize_audio, persist_upload
from app.services.diarization import DiarizationService
from app.services.transcription import TranscriptionService

router = APIRouter(
    prefix="/internal/v1/audio",
    tags=["internal-audio"],
    dependencies=[Depends(verify_internal_token)],
    responses=INTERNAL_ERROR_RESPONSES,
)


@router.post("/analyze", response_model=AnalyzeAudioResponse, status_code=200)
async def analyze_audio(
    request: Request,
    audio: UploadFile = File(...),
    source_audio_file_id: UUID = Form(..., alias="sourceAudioFileId"),
    settings: Settings = Depends(get_settings),
    transcription_service: TranscriptionService = Depends(get_transcription_service),
    diarization_service: DiarizationService = Depends(get_diarization_service),
) -> AnalyzeAudioResponse:
    form = await request.form()
    if set(form.keys()) != {"audio", "sourceAudioFileId"}:
        raise HTTPException(status_code=422, detail={"code": "INVALID_INPUT"})
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=422, detail={"code": "INVALID_INPUT"})

    job_dir = settings.tmp_dir / f"internal-analyze-{uuid4()}"
    try:
        uploaded_path = await persist_upload(audio, job_dir)
        normalized_path = job_dir / "normalized-16k-mono.wav"
        normalize_audio(uploaded_path, normalized_path, settings)
        transcript, utterances = await transcription_service.transcribe(normalized_path)
        diarization_available, segments = diarization_service.analyze(normalized_path)
        if transcript.provider == "none" or not diarization_available:
            raise InferenceFailure(InferenceFailureCode.UNAVAILABLE)

        mapped_utterances = []
        for utterance in utterances:
            speaker_label, _ = find_best_overlap(utterance, segments)
            speaker_label = speaker_label or utterance.deepgramSpeaker
            if speaker_label is None or utterance.endSec < utterance.startSec:
                raise InferenceFailure(InferenceFailureCode.INVALID_RESPONSE)
            mapped_utterances.append(
                AudioUtterance(
                    speaker_label=speaker_label,
                    started_at_ms=round(utterance.startSec * 1000),
                    ended_at_ms=round(utterance.endSec * 1000),
                    text=utterance.transcript,
                    confidence=utterance.confidence,
                    source_audio_file_id=source_audio_file_id,
                )
            )

        speakers = [
            AudioDiarizedSpeaker(speaker_label=label, candidates=[])
            for label in sorted({segment.diarized_speaker for segment in segments})
        ]
        return AnalyzeAudioResponse(
            source_audio_file_id=source_audio_file_id,
            transcript=AudioTranscript(
                provider=transcript.provider,
                model=transcript.model,
                language=transcript.language,
                text=transcript.text,
                confidence=transcript.confidence,
            ),
            utterances=mapped_utterances,
            diarized_speakers=speakers,
        )
    except InferenceFailure:
        raise
    except httpx.TimeoutException as error:
        raise InferenceFailure(InferenceFailureCode.TIMEOUT) from error
    except httpx.HTTPStatusError as error:
        code = (
            InferenceFailureCode.RATE_LIMITED
            if error.response.status_code == 429
            else InferenceFailureCode.UNAVAILABLE
        )
        raise InferenceFailure(code) from error
    except Exception as error:
        raise InferenceFailure(InferenceFailureCode.UNAVAILABLE) from error
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
