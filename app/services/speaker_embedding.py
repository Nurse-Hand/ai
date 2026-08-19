import math
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from app.services.audio import concat_segments, read_audio_mono


class SpeakerEmbeddingService:
    backend_name = "mfcc_mean_std"

    def extract_embedding(
        self,
        audio_path: Path,
        segments: Optional[list[tuple[float, float]]] = None,
    ) -> list[float]:
        audio, sample_rate = read_audio_mono(audio_path)
        clipped = concat_segments(audio, sample_rate, segments or [])
        if clipped.size == 0:
            raise ValueError("No voiced samples found for speaker embedding.")

        mfcc = librosa.feature.mfcc(y=clipped, sr=sample_rate, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        features = np.concatenate(
            [
                mfcc.mean(axis=1),
                mfcc.std(axis=1),
                delta.mean(axis=1),
                delta.std(axis=1),
            ]
        )
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return [float(value) for value in features]

    def cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        left_vec = np.array(left, dtype=np.float32)
        right_vec = np.array(right, dtype=np.float32)
        denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
        if math.isclose(denom, 0.0):
            return 0.0
        return float(np.dot(left_vec, right_vec) / denom)
