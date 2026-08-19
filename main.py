from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI

from app.config import Settings, get_settings
from app.deps import get_speaker_store
from app.routers import diarization, handoffs, speakers, tasks
from app.services.speaker_store import SpeakerStore

app = FastAPI(title="Nurse Hand AI Server")
app.include_router(tasks.router)
app.include_router(handoffs.router)
app.include_router(diarization.router)
app.include_router(speakers.router)


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
