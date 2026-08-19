from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from app.contract_base import StrictCamelModel

BoundedText = Annotated[str, StringConstraints(min_length=1, max_length=1000)]
CandidateKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")]
TaskPriority = Literal["CRITICAL", "HIGH", "NORMAL"]
TaskAiConfidence = Literal["HIGH", "MEDIUM", "LOW"]
EvidenceSourceType = Literal["TIMELINE_EVENT", "TASK"]


class TaskExtractionEvidence(StrictCamelModel):
    record_id: UUID
    source_type: EvidenceSourceType
    source_id: UUID
    patient_id: UUID | None
    work_date: AwareDatetime
    summary: BoundedText


class ExtractTasksRequest(StrictCamelModel):
    request_id: UUID
    evidence: list[TaskExtractionEvidence] = Field(max_length=1000)

    @model_validator(mode="after")
    def validate_unique_sources(self):
        source_ids = [item.source_id for item in self.evidence]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("evidence sourceId must be unique")
        return self


class ExtractedTaskCandidate(StrictCamelModel):
    candidate_key: CandidateKey
    patient_id: UUID | None
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    description: Annotated[str, StringConstraints(max_length=1000)] | None
    due_at: AwareDatetime | None
    evidence_source_ids: list[UUID] = Field(min_length=1, max_length=1000)
    confidence: TaskAiConfidence

    @model_validator(mode="after")
    def validate_unique_evidence(self):
        if len(self.evidence_source_ids) != len(set(self.evidence_source_ids)):
            raise ValueError("evidenceSourceIds must be unique")
        return self


class ExtractTasksResponse(StrictCamelModel):
    request_id: UUID
    candidates: list[ExtractedTaskCandidate] = Field(max_length=50)


class PrioritizeTasksRequest(StrictCamelModel):
    request_id: UUID
    candidates: list[ExtractedTaskCandidate] = Field(max_length=50)

    @model_validator(mode="after")
    def validate_unique_candidates(self):
        keys = [item.candidate_key for item in self.candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("candidateKey must be unique")
        return self


class TaskPrioritySuggestion(StrictCamelModel):
    candidate_key: CandidateKey
    suggested_priority: TaskPriority
    reasons: list[Annotated[str, StringConstraints(min_length=1, max_length=500)]] = Field(
        min_length=1, max_length=20
    )
    confidence: TaskAiConfidence
    evidence_source_ids: list[UUID] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_unique_evidence(self):
        if len(self.evidence_source_ids) != len(set(self.evidence_source_ids)):
            raise ValueError("evidenceSourceIds must be unique")
        return self


class PrioritizeTasksResponse(StrictCamelModel):
    request_id: UUID
    suggestions: list[TaskPrioritySuggestion] = Field(max_length=50)
