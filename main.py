from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.deps import get_speaker_store
from app.routers import diarization, handoffs, schedule_ocr, speakers, tasks
from app.routers.schedule_ocr import schedule_ocr_error_handler
from app.schedule_ocr.errors import ScheduleOcrError
from app.services.speaker_store import SpeakerStore

app = FastAPI(title="Nurse Hand AI Server")
app.include_router(tasks.router)
app.include_router(handoffs.router)
app.include_router(diarization.router)
app.include_router(speakers.router)
app.include_router(schedule_ocr.router)
app.add_exception_handler(ScheduleOcrError, schedule_ocr_error_handler)

# ponytail: dev-dashboard.html(로컬 파일)에서 브라우저 fetch로 테스트하기 위한 CORS 허용.
# 전체 오픈이라 실제 배포 전엔 백엔드 origin만 허용하도록 좁혀야 함.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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
