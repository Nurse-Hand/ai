from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, StringConstraints

from app.contract_base import StrictCamelModel

Text1000 = Annotated[str, StringConstraints(min_length=1, max_length=1000)]
SourceType = Literal["TIMELINE_EVENT", "TASK"]
ClinicalSection = Literal[
    "PATIENT_STATUS", "PAIN", "TREATMENT", "DIET", "ACTIVITY", "OBSERVATION"
]
Severity = Literal["CRITICAL", "RECOMMENDED"]


class TimelineEvent(StrictCamelModel):
    id: UUID
    occurred_at: AwareDatetime
    type: Literal["OBSERVATION", "MEDICATION", "PROCEDURE", "REPORT", "TASK"]
    summary: Text1000
    source_reference: Text1000


class SnapshotTask(StrictCamelModel):
    id: UUID
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    due_at: AwareDatetime | None
    effective_priority: Literal["CRITICAL", "HIGH", "NORMAL"]
    version: int = Field(ge=0)
    source_references: list[Text1000] = Field(max_length=1000)


class HandoffPatientSnapshot(StrictCamelModel):
    patient_id: UUID
    timeline_events: list[TimelineEvent] = Field(max_length=1000)
    tasks: list[SnapshotTask] = Field(max_length=1000)


class EvidenceReference(StrictCamelModel):
    source_type: SourceType
    source_id: UUID
    patient_id: UUID


class HandoffPrecheckRequest(StrictCamelModel):
    request_id: UUID
    patients: list[HandoffPatientSnapshot] = Field(max_length=1000)


class HandoffQuestion(StrictCamelModel):
    question_key: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    patient_id: UUID
    severity: Severity
    prompt: Text1000
    reason: Text1000
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=1000)


class HandoffPrecheckResponse(StrictCamelModel):
    request_id: UUID
    model_version: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")]
    contract_version: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")]
    generated_at: AwareDatetime
    questions: list[HandoffQuestion] = Field(max_length=1000)


class HandoffPrecheckItem(StrictCamelModel):
    id: UUID
    severity: Severity
    question: Text1000
    answer: Literal["NO_ISSUE", "INCLUDE_HANDOFF", "UNVERIFIED", "NOT_APPLICABLE"] | None
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=1000)


class GenerateHandoffRequest(StrictCamelModel):
    request_id: UUID
    template_id: Literal["NURSING_HANDOFF_V1"]
    include_unverified: bool
    patients: list[HandoffPatientSnapshot] = Field(max_length=1000)
    precheck_items: list[HandoffPrecheckItem] = Field(max_length=1000)


class HandoffSection(StrictCamelModel):
    section: ClinicalSection
    content: Annotated[str, StringConstraints(min_length=1, max_length=5000)]
    citations: list[EvidenceReference] = Field(max_length=1000)


class HandoffPatientDraft(StrictCamelModel):
    patient_id: UUID
    sections: list[HandoffSection] = Field(min_length=6, max_length=6)


class HandoffWarning(StrictCamelModel):
    code: Literal["UNVERIFIED_INFORMATION"]
    item_id: UUID
    patient_id: UUID
    message: Text1000
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=1000)


class GenerateHandoffResponse(StrictCamelModel):
    request_id: UUID
    model_version: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")]
    contract_version: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")]
    generated_at: AwareDatetime
    patients: list[HandoffPatientDraft] = Field(max_length=1000)
    warnings: list[HandoffWarning] = Field(max_length=1000)
