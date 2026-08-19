import asyncio
from io import BytesIO
from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers, UploadFile

from app.deps import get_diarization_service, get_transcription_service
from app.routers import audio as audio_router
from app.schemas import ServerTranscript, ServerTranscriptUtterance
from app.services.audio import (
    AudioDecodeError,
    AudioProcessingTimeoutError,
    AudioTooLargeError,
    cleanup_job_dir,
    persist_upload,
)
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


async def fake_persist(_upload, dest_dir, max_bytes=None):
    assert max_bytes is not None
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


def test_streaming_upload_limit_is_enforced_before_excess_bytes_are_written(tmp_path):
    upload = UploadFile(
        file=BytesIO(b"12345"),
        filename="clinical-raw-name.wav",
        headers=Headers({"content-type": "audio/wav"}),
    )
    job_dir = tmp_path / "job"
    with pytest.raises(AudioTooLargeError):
        asyncio.run(persist_upload(upload, job_dir, max_bytes=4))
    written = list(job_dir.iterdir())
    assert len(written) == 1
    assert written[0].stat().st_size == 0
    assert "clinical-raw-name" not in written[0].name
    assert cleanup_job_dir(job_dir, attempts=2)


def test_request_size_is_rejected_before_audio_services(client, auth_headers, monkeypatch):
    called = False

    async def should_not_persist(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(audio_router, "persist_upload", should_not_persist)
    response = client.post(
        "/internal/v1/audio/analyze",
        headers={**auth_headers, "Content-Length": str(30 * 1024 * 1024)},
        data={"sourceAudioFileId": SOURCE_FILE_ID},
        files={"audio": ("synthetic.wav", b"synthetic", "audio/wav")},
    )
    assert response.status_code == 422
    assert not called


def test_decode_failure_is_invalid_input_without_raw_log(
    client, auth_headers, monkeypatch, caplog
):
    install_fakes(monkeypatch)
    monkeypatch.setattr(
        audio_router,
        "normalize_audio",
        lambda *_args: (_ for _ in ()).throw(AudioDecodeError("SENSITIVE_RAW_AUDIO")),
    )
    response = client.post(
        "/internal/v1/audio/analyze",
        headers=auth_headers,
        data={"sourceAudioFileId": SOURCE_FILE_ID},
        files={
            "audio": (
                "patient-clinical-recording.wav",
                b"SENSITIVE_RAW_AUDIO",
                "audio/wav",
            )
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_INPUT"
    assert "patient-clinical-recording" not in caplog.text
    assert "SENSITIVE_RAW_AUDIO" not in caplog.text


def test_non_audio_mime_is_invalid_input(client, auth_headers):
    response = client.post(
        "/internal/v1/audio/analyze",
        headers=auth_headers,
        data={"sourceAudioFileId": SOURCE_FILE_ID},
        files={"audio": ("synthetic.bin", b"synthetic", "application/octet-stream")},
    )
    assert response.status_code == 422


class TimeoutTranscriptionService:
    async def transcribe(self, _path):
        raise asyncio.TimeoutError


def test_audio_timeout_maps_to_504(client, auth_headers, monkeypatch):
    install_fakes(monkeypatch)
    app.dependency_overrides[get_transcription_service] = lambda: TimeoutTranscriptionService()
    response = client.post(
        "/internal/v1/audio/analyze",
        headers=auth_headers,
        data={"sourceAudioFileId": SOURCE_FILE_ID},
        files={"audio": ("synthetic.wav", b"synthetic", "audio/wav")},
    )
    assert response.status_code == 504


def test_ffmpeg_timeout_maps_to_504(client, auth_headers, monkeypatch):
    install_fakes(monkeypatch)
    monkeypatch.setattr(
        audio_router,
        "normalize_audio",
        lambda *_args: (_ for _ in ()).throw(AudioProcessingTimeoutError()),
    )
    response = client.post(
        "/internal/v1/audio/analyze",
        headers=auth_headers,
        data={"sourceAudioFileId": SOURCE_FILE_ID},
        files={"audio": ("synthetic.wav", b"synthetic", "audio/wav")},
    )
    assert response.status_code == 504


class TimeoutDiarizationService:
    def analyze(self, _path):
        raise asyncio.TimeoutError


def test_diarization_timeout_maps_to_504(client, auth_headers, monkeypatch):
    install_fakes(monkeypatch)
    app.dependency_overrides[get_diarization_service] = lambda: TimeoutDiarizationService()
    response = client.post(
        "/internal/v1/audio/analyze",
        headers=auth_headers,
        data={"sourceAudioFileId": SOURCE_FILE_ID},
        files={"audio": ("synthetic.wav", b"synthetic", "audio/wav")},
    )
    assert response.status_code == 504


class MalformedTranscriptionService:
    async def transcribe(self, _path):
        return (
            SimpleNamespace(
                provider="test-stt",
                model="test-model",
                language="ko-KR",
                text="synthetic",
                confidence="0.9",
            ),
            [],
        )


def test_malformed_audio_upstream_maps_to_502(client, auth_headers, monkeypatch):
    install_fakes(monkeypatch)
    app.dependency_overrides[get_transcription_service] = lambda: MalformedTranscriptionService()
    response = client.post(
        "/internal/v1/audio/analyze",
        headers=auth_headers,
        data={"sourceAudioFileId": SOURCE_FILE_ID},
        files={"audio": ("synthetic.wav", b"synthetic", "audio/wav")},
    )
    assert response.status_code == 502


def test_cleanup_failure_is_fail_closed_and_safely_logged(
    client, auth_headers, monkeypatch, caplog
):
    install_fakes(monkeypatch)
    monkeypatch.setattr(audio_router, "cleanup_job_dir", lambda *_args: False)
    response = client.post(
        "/internal/v1/audio/analyze",
        headers=auth_headers,
        data={"sourceAudioFileId": SOURCE_FILE_ID},
        files={"audio": ("clinical-secret.wav", b"RAW_SECRET", "audio/wav")},
    )
    assert response.status_code == 503
    assert "Internal audio temporary cleanup failed." in caplog.text
    assert "clinical-secret" not in caplog.text
    assert "RAW_SECRET" not in caplog.text
