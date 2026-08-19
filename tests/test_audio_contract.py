import asyncio
from contextlib import suppress
from io import BytesIO
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers, UploadFile

from app.config import Settings, get_settings
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


class DelayedBlockingDiarizationService:
    def __init__(self, started, release, finished):
        self.started = started
        self.release = release
        self.finished = finished

    def analyze(self, _path):
        self.started.set()
        self.release.wait(timeout=1)
        self.finished.set()
        raise RuntimeError("SENSITIVE_RAW_PATH_AND_PAYLOAD")


def test_blocking_diarization_timeout_tracks_worker_before_cleanup(
    client, auth_headers, monkeypatch, tmp_path, caplog
):
    install_fakes(monkeypatch)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    service = DelayedBlockingDiarizationService(started, release, finished)
    settings = Settings(
        internal_token="test-internal-token",
        tmp_dir=tmp_path,
        audio_processing_timeout_seconds=0.01,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_diarization_service] = lambda: service

    real_cleanup = audio_router.cleanup_job_dir
    cleanup_observations = []
    cleanup_finished = threading.Event()

    def guarded_cleanup(job_dir, attempts):
        cleanup_observations.append(finished.is_set())
        result = real_cleanup(job_dir, attempts)
        cleanup_finished.set()
        return result

    monkeypatch.setattr(audio_router, "cleanup_job_dir", guarded_cleanup)
    started_at = time.perf_counter()
    response = client.post(
        "/internal/v1/audio/analyze",
        headers=auth_headers,
        data={"sourceAudioFileId": SOURCE_FILE_ID},
        files={
            "audio": (
                "clinical-raw-name.wav",
                b"SENSITIVE_RAW_AUDIO",
                "audio/wav",
            )
        },
    )
    elapsed = time.perf_counter() - started_at

    assert started.is_set()
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "AI_UPSTREAM_TIMEOUT"
    assert elapsed < 0.2
    assert not finished.is_set()
    assert cleanup_observations == []
    assert len(list(tmp_path.glob("internal-analyze-*"))) == 1

    asyncio.run(audio_router.drain_audio_workers(timeout_seconds=0.001))

    async def cancel_shutdown_drain():
        drain = asyncio.create_task(audio_router.drain_audio_workers(timeout_seconds=5))
        await asyncio.sleep(0)
        drain.cancel()
        with suppress(asyncio.CancelledError):
            await drain

    asyncio.run(cancel_shutdown_drain())
    assert not finished.is_set()
    assert cleanup_observations == []
    assert len(list(tmp_path.glob("internal-analyze-*"))) == 1

    release.set()
    assert finished.wait(timeout=1)
    assert cleanup_finished.wait(timeout=1)
    assert cleanup_observations == [True]
    assert list(tmp_path.glob("internal-analyze-*")) == []
    assert "Internal audio background worker failed." in caplog.text
    assert "Internal audio temporary cleanup failed." not in caplog.text
    assert "clinical-raw-name" not in caplog.text
    assert "SENSITIVE_RAW_AUDIO" not in caplog.text
    assert "SENSITIVE_RAW_PATH_AND_PAYLOAD" not in caplog.text


class SaturatingDiarizationService:
    def __init__(self, capacity):
        self.capacity = capacity
        self.release = threading.Event()
        self.all_finished = threading.Event()
        self.lock = threading.Lock()
        self.started = 0
        self.finished = 0
        self.daemon_flags = []

    def analyze(self, _path):
        with self.lock:
            self.started += 1
            self.daemon_flags.append(threading.current_thread().daemon)
        self.release.wait(timeout=2)
        with self.lock:
            self.finished += 1
            if self.finished == self.capacity:
                self.all_finished.set()
        return True, [RawSegment(0.0, 1.25, "SPEAKER_00")]


def test_audio_worker_capacity_has_no_queue_and_recovers_permits(
    client, auth_headers, monkeypatch, tmp_path
):
    assert not hasattr(audio_router, "_diarization_executor")
    assert audio_router._audio_worker_count() == 0
    capacity = audio_router._audio_worker_capacity
    service = SaturatingDiarizationService(capacity)
    settings = Settings(
        internal_token="test-internal-token",
        tmp_dir=tmp_path,
        audio_processing_timeout_seconds=0.01,
    )
    install_fakes(monkeypatch)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_diarization_service] = lambda: service

    real_cleanup = audio_router.cleanup_job_dir
    tracking_during_cleanup = []

    def observed_cleanup(job_dir, attempts):
        tracking_during_cleanup.append(audio_router._audio_worker_count())
        return real_cleanup(job_dir, attempts)

    monkeypatch.setattr(audio_router, "cleanup_job_dir", observed_cleanup)
    for expected in range(1, capacity + 1):
        response = client.post(
            "/internal/v1/audio/analyze",
            headers=auth_headers,
            data={"sourceAudioFileId": SOURCE_FILE_ID},
            files={"audio": ("synthetic.wav", b"synthetic", "audio/wav")},
        )
        assert response.status_code == 504
        assert audio_router._audio_worker_count() == expected

    before_dirs = set(tmp_path.glob("internal-analyze-*"))
    assert len(before_dirs) == capacity
    started_at = time.perf_counter()
    rejected = client.post(
        "/internal/v1/audio/analyze",
        headers=auth_headers,
        data={"sourceAudioFileId": SOURCE_FILE_ID},
        files={"audio": ("rejected.wav", b"rejected", "audio/wav")},
    )
    assert time.perf_counter() - started_at < 0.2
    assert rejected.status_code == 503
    assert rejected.json()["error"]["code"] == "AI_UPSTREAM_UNAVAILABLE"
    assert set(tmp_path.glob("internal-analyze-*")) == before_dirs
    assert audio_router._audio_worker_count() == capacity

    started_at = time.perf_counter()
    asyncio.run(audio_router.drain_audio_workers(timeout_seconds=0.001))
    assert time.perf_counter() - started_at < 0.2
    assert audio_router._audio_worker_count() == capacity
    assert set(tmp_path.glob("internal-analyze-*")) == before_dirs

    service.release.set()
    assert service.all_finished.wait(timeout=1)
    asyncio.run(audio_router.drain_audio_workers(timeout_seconds=1))
    assert audio_router._audio_worker_count() == 0
    assert list(tmp_path.glob("internal-analyze-*")) == []
    assert len(tracking_during_cleanup) == capacity
    assert all(count >= 1 for count in tracking_during_cleanup)
    assert service.daemon_flags == [True] * capacity

    settings.audio_processing_timeout_seconds = 0.5
    app.dependency_overrides[get_diarization_service] = lambda: FakeDiarizationService()
    recovered = client.post(
        "/internal/v1/audio/analyze",
        headers=auth_headers,
        data={"sourceAudioFileId": SOURCE_FILE_ID},
        files={"audio": ("synthetic.wav", b"synthetic", "audio/wav")},
    )
    assert recovered.status_code == 200
    assert audio_router._audio_worker_count() == 0


def test_container_tmp_contract_is_non_persistent():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "VOLUME" not in dockerfile.upper()
    assert "host volume mount 금지" in readme
    assert Settings(_env_file=None).audio_worker_capacity == 4


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
    assert "Internal audio background cleanup failed." in caplog.text
    assert "clinical-secret" not in caplog.text
    assert "RAW_SECRET" not in caplog.text
