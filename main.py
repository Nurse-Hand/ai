from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.cors import parse_cors_allowed_origins
from app.deps import get_speaker_store
from app.routers import diarization, handoffs, schedule_ocr, speakers, tasks
from app.routers.schedule_ocr import schedule_ocr_error_handler
from app.schedule_ocr.body_limit import ScheduleOcrBodyLimitMiddleware
from app.schedule_ocr.errors import ScheduleOcrError
from app.services.speaker_store import SpeakerStore

app = FastAPI(title="Nurse Hand AI Server")
app.include_router(tasks.router)
app.include_router(handoffs.router)
app.include_router(diarization.router)
app.include_router(speakers.router)
app.include_router(schedule_ocr.router)
app.add_exception_handler(ScheduleOcrError, schedule_ocr_error_handler)
app.add_middleware(ScheduleOcrBodyLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_allowed_origins(os.getenv("CORS_ALLOWED_ORIGINS")),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Internal-Token"],
)


@app.get("/health")
def health(settings: Settings = Depends(get_settings), speaker_store: SpeakerStore = Depends(get_speaker_store)):
    speakers_list = speaker_store.list()
    return {
        "ok": True,
        "deepgramConfigured": bool(settings.deepgram_api_key),
        "deepgramModel": settings.deepgram_model,
        "deepgramLanguage": settings.deepgram_language,
        "localSttModelDir": str(settings.local_stt_model_dir) if settings.local_stt_model_dir else None,
        "pyannoteConfigured": bool(settings.pyannote_auth_token),
        "speakerEmbeddingBackend": "mfcc_mean_std",
        "speakerCount": len(speakers_list),
        "dataDir": str(settings.data_dir),
        "tmpDir": str(settings.tmp_dir),
    }
