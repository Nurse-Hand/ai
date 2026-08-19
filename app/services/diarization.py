from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.services.audio import get_audio_duration_sec


@dataclass
class RawSegment:
    start_sec: float
    end_sec: float
    diarized_speaker: str

    @property
    def duration_sec(self) -> float:
        return round(self.end_sec - self.start_sec, 3)


class DiarizationService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def analyze(self, audio_path: Path) -> tuple[bool, list[RawSegment]]:
        segments: list[RawSegment] = []
        available = False

        try:
            if not self.settings.pyannote_auth_token:
                raise RuntimeError("PYANNOTE_AUTH_TOKEN is not set.")
            from pyannote.audio import Pipeline  # type: ignore

            pipeline = Pipeline.from_pretrained(
                self.settings.pyannote_diarization_model,
                token=self.settings.pyannote_auth_token,
            )
            output = pipeline(str(audio_path))
            # pyannote.audio 4.x: Pipeline 호출 결과가 DiarizeOutput으로 한 단계 더 감싸짐
            # (3.x는 Annotation을 바로 반환, itertracks도 거기 있었음)
            diarization = output.speaker_diarization
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = RawSegment(
                    start_sec=round(float(turn.start), 3),
                    end_sec=round(float(turn.end), 3),
                    diarized_speaker=speaker,
                )
                if segment.duration_sec >= self.settings.min_segment_sec:
                    segments.append(segment)
            available = True
        except Exception:
            available = False

        if not segments:
            duration = get_audio_duration_sec(audio_path)
            segments = [
                RawSegment(
                    start_sec=0.0,
                    end_sec=duration,
                    diarized_speaker="SPEAKER_00",
                )
            ]

        by_speaker: dict[str, float] = defaultdict(float)
        for segment in segments:
            by_speaker[segment.diarized_speaker] += segment.duration_sec

        filtered = [
            segment
            for segment in segments
            if by_speaker[segment.diarized_speaker] >= self.settings.min_speaker_total_sec
        ]
        if filtered:
            segments = filtered

        if self.settings.max_speaker_total_sec is not None:
            capped: list[RawSegment] = []
            allowed = self.settings.max_speaker_total_sec
            running: dict[str, float] = defaultdict(float)
            for segment in segments:
                if running[segment.diarized_speaker] >= allowed:
                    continue
                capped.append(segment)
                running[segment.diarized_speaker] += segment.duration_sec
            if capped:
                segments = capped

        return available, segments

