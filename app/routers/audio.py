import asyncio
import logging
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

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
from app.services.audio import (
    AudioDecodeError,
    AudioProcessingTimeoutError,
    AudioToolUnavailableError,
    AudioTooLargeError,
    cleanup_job_dir,
    normalize_audio,
    persist_upload,
)
from app.services.diarization import DiarizationService
from app.services.transcription import TranscriptionService

logger = logging.getLogger(__name__)
_background_audio_finalizers: set[asyncio.Task[None]] = set()


async def _finalize_audio_worker(
    worker: asyncio.Task,
    job_dir,
    cleanup_attempts: int,
) -> None:
    try:
        await worker
    except asyncio.CancelledError:
        logger.warning("Internal audio background worker was cancelled.")
    except Exception:
        logger.warning("Internal audio background worker failed.")
    finally:
        if not cleanup_job_dir(job_dir, cleanup_attempts):
            logger.error("Internal audio background cleanup failed.")


def _schedule_audio_finalizer(worker: asyncio.Task, job_dir, cleanup_attempts: int) -> None:
    finalizer = asyncio.create_task(
        _finalize_audio_worker(worker, job_dir, cleanup_attempts)
    )
    _background_audio_finalizers.add(finalizer)
    finalizer.add_done_callback(_background_audio_finalizers.discard)


async def drain_audio_finalizers(timeout_seconds: float = 5.0) -> None:
    pending = tuple(_background_audio_finalizers)
    if not pending:
        return
    _, still_pending = await asyncio.wait(pending, timeout=timeout_seconds)
    if still_pending:
        logger.warning("Internal audio background cleanup is still pending at shutdown.")


def enforce_audio_request_size(
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    raw_length = request.headers.get("content-length")
    try:
        content_length = int(raw_length) if raw_length is not None else None
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_INPUT"}) from error
    if content_length is None or content_length < 0:
        raise HTTPException(status_code=422, detail={"code": "INVALID_INPUT"})
    if content_length > settings.audio_max_request_bytes:
        raise HTTPException(status_code=422, detail={"code": "INVALID_INPUT"})


router = APIRouter(
    prefix="/internal/v1/audio",
    tags=["internal-audio"],
    dependencies=[Depends(verify_internal_token), Depends(enforce_audio_request_size)],
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
    diarization_worker: asyncio.Task | None = None
    cleanup_deferred = False
    try:
        uploaded_path = await persist_upload(
            audio, job_dir, max_bytes=settings.audio_max_upload_bytes
        )
        normalized_path = job_dir / "normalized-16k-mono.wav"
        normalize_audio(uploaded_path, normalized_path, settings)
        transcript, utterances = await asyncio.wait_for(
            transcription_service.transcribe(normalized_path),
            timeout=settings.audio_processing_timeout_seconds,
        )
        diarization_worker = asyncio.create_task(
            asyncio.to_thread(diarization_service.analyze, normalized_path)
        )
        try:
            diarization_available, segments = await asyncio.wait_for(
                asyncio.shield(diarization_worker),
                timeout=settings.audio_processing_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            if not diarization_worker.done():
                _schedule_audio_finalizer(
                    diarization_worker, job_dir, settings.audio_cleanup_attempts
                )
                cleanup_deferred = True
            raise InferenceFailure(InferenceFailureCode.TIMEOUT) from error
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
    except (AudioTooLargeError, AudioDecodeError) as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_INPUT"}) from error
    except (asyncio.TimeoutError, AudioProcessingTimeoutError, httpx.TimeoutException) as error:
        raise InferenceFailure(InferenceFailureCode.TIMEOUT) from error
    except ValidationError as error:
        raise InferenceFailure(InferenceFailureCode.INVALID_RESPONSE) from error
    except AudioToolUnavailableError as error:
        raise InferenceFailure(InferenceFailureCode.UNAVAILABLE) from error
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
        if (
            diarization_worker is not None
            and not diarization_worker.done()
            and not cleanup_deferred
        ):
            _schedule_audio_finalizer(
                diarization_worker, job_dir, settings.audio_cleanup_attempts
            )
            cleanup_deferred = True
        if not cleanup_deferred and not cleanup_job_dir(
            job_dir, settings.audio_cleanup_attempts
        ):
            logger.error("Internal audio temporary cleanup failed.")
            raise InferenceFailure(InferenceFailureCode.UNAVAILABLE)
