import shutil
import subprocess
from pathlib import Path
from uuid import uuid4
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import UploadFile

from app.config import Settings


def safe_name(name: Optional[str]) -> str:
    if not name:
        return "audio.wav"
    return Path(name).name


async def persist_upload(upload: UploadFile, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / f"{uuid4()}-{safe_name(upload.filename)}"
    with file_path.open("wb") as buffer:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
    await upload.close()
    return file_path


def normalize_audio(input_path: Path, output_path: Path, settings: Settings) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            settings.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
    )


def read_audio_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    return audio[:, 0], sample_rate


def get_audio_duration_sec(audio_path: Path) -> float:
    audio, sample_rate = read_audio_mono(audio_path)
    return round(float(len(audio)) / float(sample_rate), 3) if sample_rate else 0.0


def concat_segments(audio: np.ndarray, sample_rate: int, segments: list[tuple[float, float]]) -> np.ndarray:
    if not segments:
        return audio
    pieces: list[np.ndarray] = []
    for start_sec, end_sec in segments:
        start_index = max(0, int(start_sec * sample_rate))
        end_index = min(len(audio), int(end_sec * sample_rate))
        if end_index > start_index:
            pieces.append(audio[start_index:end_index])
    if not pieces:
        return np.array([], dtype=np.float32)
    return np.concatenate(pieces).astype(np.float32)


def copy_sample(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
