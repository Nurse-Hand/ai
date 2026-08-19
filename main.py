from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.deps import get_speaker_store
from app.errors import InferenceFailure, InferenceFailureCode
from app.routers import audio, diarization, handoffs, speakers, tasks
from app.services.speaker_store import SpeakerStore


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await audio.drain_audio_workers()


app = FastAPI(title="Nurse Hand AI Server", version="1.0.0", lifespan=lifespan)
app.include_router(tasks.router)
app.include_router(handoffs.router)
app.include_router(audio.router)

# Legacy external routes remain isolated until their compatibility/removal decision is made.
app.include_router(diarization.router)
app.include_router(speakers.router)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, error: RequestValidationError):
    if not request.url.path.startswith("/internal/v1/"):
        return await request_validation_exception_handler(request, error)
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "INVALID_INPUT", "message": "Request validation failed."}},
    )


@app.exception_handler(InferenceFailure)
async def inference_error_handler(_request: Request, error: InferenceFailure):
    status_by_code = {
        InferenceFailureCode.TIMEOUT: 504,
        InferenceFailureCode.RATE_LIMITED: 429,
        InferenceFailureCode.UNAVAILABLE: 503,
        InferenceFailureCode.INVALID_RESPONSE: 502,
    }
    return JSONResponse(
        status_code=status_by_code[error.code],
        content={"error": {"code": error.code.value, "message": "AI inference failed."}},
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, error: HTTPException):
    if not request.url.path.startswith("/internal/v1/"):
        return await http_exception_handler(request, error)
    detail = error.detail if isinstance(error.detail, dict) else {"code": "HTTP_ERROR"}
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": detail.get("code", "HTTP_ERROR"), "message": "Request failed."}},
    )


@app.get("/health")
def health(
    settings: Settings = Depends(get_settings),
    speaker_store: SpeakerStore = Depends(get_speaker_store),
):
    speakers_list = speaker_store.list()
    return {
        "ok": True,
        "deepgramConfigured": bool(settings.deepgram_api_key),
        "deepgramModel": settings.deepgram_model,
        "deepgramLanguage": settings.deepgram_language,
        "localSttConfigured": bool(settings.local_stt_model_dir),
        "pyannoteConfigured": bool(settings.pyannote_auth_token),
        "speakerEmbeddingBackend": "mfcc_mean_std",
        "speakerCount": len(speakers_list),
    }
