import asyncio
from concurrent.futures import Future
import logging
import threading
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
_audio_worker_capacity = get_settings().audio_worker_capacity
_audio_worker_slots = threading.BoundedSemaphore(_audio_worker_capacity)
_audio_workers: set[Future] = set()
_audio_workers_lock = threading.Lock()


def _audio_worker_count() -> int:
    with _audio_workers_lock:
        return len(_audio_workers)


def _try_accept_audio_analysis() -> Future | None:
    if not _audio_worker_slots.acquire(blocking=False):
        return None
    worker: Future = Future()
    with _audio_workers_lock:
        _audio_workers.add(worker)
    return worker


def _run_diarization_worker(worker: Future, diarization_service, normalized_path) -> None:
    if not worker.set_running_or_notify_cancel():
        return
    try:
        result = diarization_service.analyze(normalized_path)
    except BaseException as error:
        worker.set_exception(error)
    else:
        worker.set_result(result)


def _start_diarization_worker(
    worker: Future, diarization_service, normalized_path
) -> bool:
    thread = threading.Thread(
        target=_run_diarization_worker,
        args=(worker, diarization_service, normalized_path),
        name="internal-audio-diarization",
        daemon=True,
    )
    try:
        thread.start()
    except Exception as error:
        worker.set_exception(error)
        return False
    return True


def _consume_asyncio_worker_result(worker: asyncio.Future) -> None:
    if worker.cancelled():
        return
    try:
        worker.exception()
    except BaseException:
        pass


def _complete_audio_analysis(
    worker: Future,
    job_dir,
    cleanup_attempts: int,
    cleanup_state: dict[str, bool],
) -> None:
    try:
        try:
            worker.result()
        except BaseException:
            logger.warning("Internal audio background worker failed.")
        try:
            cleanup_state["ok"] = cleanup_job_dir(job_dir, cleanup_attempts)
        except Exception:
            cleanup_state["ok"] = False
        if not cleanup_state["ok"]:
            logger.error("Internal audio background cleanup failed.")
    finally:
        with _audio_workers_lock:
            _audio_workers.discard(worker)
        _audio_worker_slots.release()


def _attach_audio_completion(
    worker: Future,
    job_dir,
    cleanup_attempts: int,
    cleanup_state: dict[str, bool],
) -> None:
    worker.add_done_callback(
        lambda completed: _complete_audio_analysis(
            completed,
            job_dir,
            cleanup_attempts,
            cleanup_state,
        )
    )


async def drain_audio_workers(timeout_seconds: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout_seconds)
    while _audio_worker_count():
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.warning("Internal audio background cleanup is still pending at shutdown.")
            return
        await asyncio.sleep(min(0.05, remaining))


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

    analysis_worker = _try_accept_audio_analysis()
    if analysis_worker is None:
        raise InferenceFailure(InferenceFailureCode.UNAVAILABLE)

    job_dir = settings.tmp_dir / f"internal-analyze-{uuid4()}"
    worker_started = False
    completion_attached = False
    cleanup_state: dict[str, bool] = {}
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
        worker_started = _start_diarization_worker(
            analysis_worker, diarization_service, normalized_path
        )
        wrapped_worker = asyncio.wrap_future(analysis_worker)
        wrapped_worker.add_done_callback(_consume_asyncio_worker_result)
        try:
            diarization_available, segments = await asyncio.wait_for(
                asyncio.shield(wrapped_worker),
                timeout=settings.audio_processing_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            if not analysis_worker.done():
                _attach_audio_completion(
                    analysis_worker,
                    job_dir,
                    settings.audio_cleanup_attempts,
                    cleanup_state,
                )
                completion_attached = True
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
        if not completion_attached:
            _attach_audio_completion(
                analysis_worker,
                job_dir,
                settings.audio_cleanup_attempts,
                cleanup_state,
            )
            completion_attached = True
        if not worker_started and not analysis_worker.done():
            analysis_worker.set_result(None)
        if cleanup_state.get("ok") is False:
            raise InferenceFailure(InferenceFailureCode.UNAVAILABLE)
