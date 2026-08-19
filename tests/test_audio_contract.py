from app.deps import get_diarization_service, get_transcription_service
from app.routers import audio as audio_router
from app.schemas import ServerTranscript, ServerTranscriptUtterance
from app.services.diarization import RawSegment
from main import app

SOURCE_FILE_ID = "00000000-0000-4000-8000-000000000021"


class FakeTranscriptionService:
    async def transcribe(self, _path):
        return (
            ServerTranscript(
                provider="test-stt",
                model="test-model",
                language="ko-KR",
                text="synthetic transcript",
                confidence=0.9,
            ),
            [
                ServerTranscriptUtterance(
                    utteranceId="utt-1",
                    startSec=0.0,
                    endSec=1.25,
                    durationSec=1.25,
                    transcript="synthetic utterance",
                    confidence=0.8,
                    deepgramSpeaker="SPEAKER_00",
                    bestMatch=None,
                    candidates=[],
                )
            ],
        )


class FakeDiarizationService:
    def analyze(self, _path):
        return True, [RawSegment(0.0, 1.25, "SPEAKER_00")]


async def fake_persist(_upload, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / "synthetic.wav"
    path.write_bytes(b"synthetic")
    return path


def install_fakes(monkeypatch):
    app.dependency_overrides[get_transcription_service] = lambda: FakeTranscriptionService()
    app.dependency_overrides[get_diarization_service] = lambda: FakeDiarizationService()
    monkeypatch.setattr(audio_router, "persist_upload", fake_persist)
    monkeypatch.setattr(audio_router, "normalize_audio", lambda _source, _target, _settings: None)


def test_internal_audio_analyze_returns_transcript_utterances_and_candidates(
    client, auth_headers, monkeypatch
):
    install_fakes(monkeypatch)
    response = client.post(
        "/internal/v1/audio/analyze",
        headers=auth_headers,
        data={"sourceAudioFileId": SOURCE_FILE_ID},
        files={"audio": ("synthetic.wav", b"synthetic", "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["sourceAudioFileId"] == SOURCE_FILE_ID
    assert payload["transcript"]["text"] == "synthetic transcript"
    assert payload["utterances"] == [
        {
            "speakerLabel": "SPEAKER_00",
            "startedAtMs": 0,
            "endedAtMs": 1250,
            "text": "synthetic utterance",
            "confidence": 0.8,
            "sourceAudioFileId": SOURCE_FILE_ID,
        }
    ]
    assert payload["diarizedSpeakers"] == [{"speakerLabel": "SPEAKER_00", "candidates": []}]
    assert "patientId" not in str(payload)
    assert "bestMatch" not in str(payload)


def test_internal_audio_requires_auth(client):
    response = client.post(
        "/internal/v1/audio/analyze",
        data={"sourceAudioFileId": SOURCE_FILE_ID},
        files={"audio": ("synthetic.wav", b"synthetic", "audio/wav")},
    )
    assert response.status_code == 401


def test_internal_audio_rejects_unknown_form_field(client, auth_headers, monkeypatch):
    install_fakes(monkeypatch)
    response = client.post(
        "/internal/v1/audio/analyze",
        headers=auth_headers,
        data={"sourceAudioFileId": SOURCE_FILE_ID, "unknown": "value"},
        files={"audio": ("synthetic.wav", b"synthetic", "audio/wav")},
    )
    assert response.status_code == 422


def test_audio_response_rejects_string_confidence():
    from pydantic import ValidationError

    from app.audio_contracts import AudioTranscript

    try:
        AudioTranscript.model_validate(
            {
                "provider": "test",
                "model": "test",
                "language": "ko-KR",
                "text": "synthetic",
                "confidence": "0.9",
            }
        )
    except ValidationError:
        return
    raise AssertionError("string confidence must be rejected")
