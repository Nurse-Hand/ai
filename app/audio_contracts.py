from typing import Annotated
from uuid import UUID

from pydantic import Field, StrictFloat, StrictInt, StringConstraints

from app.contract_base import StrictCamelModel


class AudioTranscript(StrictCamelModel):
    provider: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    model: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    language: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    text: Annotated[str, StringConstraints(max_length=1000000)]
    confidence: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)


class AudioUtterance(StrictCamelModel):
    speaker_label: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    started_at_ms: StrictInt = Field(ge=0)
    ended_at_ms: StrictInt = Field(ge=0)
    text: Annotated[str, StringConstraints(max_length=100000)]
    confidence: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    source_audio_file_id: UUID


class AudioSpeakerCandidate(StrictCamelModel):
    speaker_id: UUID
    display_name: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    similarity: StrictFloat = Field(ge=-1.0, le=1.0)


class AudioDiarizedSpeaker(StrictCamelModel):
    speaker_label: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    candidates: list[AudioSpeakerCandidate] = Field(max_length=3)


class AnalyzeAudioResponse(StrictCamelModel):
    source_audio_file_id: UUID
    transcript: AudioTranscript
    utterances: list[AudioUtterance] = Field(max_length=100000)
    diarized_speakers: list[AudioDiarizedSpeaker] = Field(max_length=1000)
