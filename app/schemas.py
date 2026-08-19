from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """백엔드↔AI 내부 API(노션 확정 스키마)는 camelCase. 파이썬 쪽은 snake_case 유지,
    직렬화/역직렬화만 alias로 자동 변환."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# ══════════════════════════════════════════════════════════════════
# 아래부터는 음성 분석(화자분리) 스키마 - backend-ai에서 이식 (2026-08-19)
# 이쪽은 이미 검증된 실제 응답 필드명 그대로라 CamelModel(자동 변환) 대신
# camelCase 필드명을 직접 씀
# ══════════════════════════════════════════════════════════════════


class DiarizationCandidate(BaseModel):
    speakerId: str
    displayName: str
    similarity: float


class DiarizationSegment(BaseModel):
    startSec: float
    endSec: float
    durationSec: float
    diarizedSpeaker: str
    bestMatch: Optional[DiarizationCandidate]
    candidates: list[DiarizationCandidate]


class ServerTranscript(BaseModel):
    provider: str
    model: str
    language: str
    text: str
    confidence: Optional[float] = None


class ServerTranscriptUtterance(BaseModel):
    utteranceId: str
    startSec: float
    endSec: float
    durationSec: float
    transcript: str
    confidence: Optional[float] = None
    deepgramSpeaker: Optional[str] = None
    diarizedSpeaker: Optional[str] = None
    overlapSeconds: Optional[float] = None
    bestMatch: Optional[DiarizationCandidate]
    candidates: list[DiarizationCandidate]
    speakerRoles: Optional[list[str]] = None
    handoverPinned: Optional[bool] = None


class DiarizationSpeakerMatch(BaseModel):
    diarizedSpeaker: str
    segmentCount: int
    totalSpeechSec: float
    embeddingAvailable: bool
    bestMatch: Optional[DiarizationCandidate]
    candidates: list[DiarizationCandidate]
    representativeQuote: Optional[ServerTranscriptUtterance]


class RoundingDiarizationAnalysis(BaseModel):
    fileName: str
    diarizationAvailable: bool
    threshold: float
    rawSegmentCount: int
    minSegmentSec: Optional[float] = None
    minSpeakerTotalSec: Optional[float] = None
    maxSpeakerTotalSec: Optional[float] = None
    totalSegments: int
    segments: list[DiarizationSegment]
    speakerMatches: list[DiarizationSpeakerMatch]
    transcript: ServerTranscript
    utterances: list[ServerTranscriptUtterance]


class SpeakerRecord(BaseModel):
    speakerId: str
    displayName: str
    registeredAt: datetime
    updatedAt: datetime
    embedding: list[float]
    samplePath: Optional[str] = None


class SpeakersListResponse(BaseModel):
    total: int
    cacheMode: str
    speakers: list[SpeakerRecord]


class DeleteSpeakerResponse(BaseModel):
    deletedSpeakerId: str
    total: int
    cacheMode: str


class RegisterDiarizedSpeakerResponse(BaseModel):
    speaker: dict[str, Any]
    diarization: dict[str, Any]
    embedding: dict[str, Any]
    cache: dict[str, Any]


class Task(CamelModel):
    """노션 확정 스키마 (`/internal/v1/tasks/prioritize`의 tasks)."""
    task_id: str
    patient_id: str
    title: str
    due_at: str | None = None
    carried_over: bool = False


class MissingItem(BaseModel):
    patient_id: str
    field_id: str
    description: str
    ai_evidence: str
    severity: Literal["중요", "권장"]


class PatientRisk(CamelModel):
    patient_id: str
    level: str  # 노션 예시 "HIGH" 하나뿐, 전체 enum 미확정


class PriorityResult(CamelModel):
    task_id: str
    score: float
    priority: str  # 노션 예시 "CRITICAL" 하나뿐, 전체 enum 미확정
    reasons: list[str] = []


# ── 요청/응답 wrapper (Backend→AI 내부 API, requestId 필수 - 노션 확정) ──

class PrioritizeTasksRequest(CamelModel):
    request_id: str
    tasks: list[Task]
    patient_risk: list[PatientRisk] = []
    now: str


class PrioritizeTasksResponse(CamelModel):
    request_id: str
    results: list[PriorityResult]


class DraftItemRef(CamelModel):
    """역검증 대상이 되는, 이미 생성된 초안(handoffDraft)의 항목 요약."""
    topic: str
    summary: str


class Evidence(CamelModel):
    """노션 '저장 로직'/'LLM 최종 제공 템플릿' 페이지 기준 evidence 단위.
    generate의 evidences와 precheck의 candidateEvidence 둘 다 이 스키마를 쓴다 - 같은
    저장 개체(evidence)를 두 플로우가 다르게 부르는 것뿐이라 예전에 따로 만든
    DraftEvidence(얕은 버전)와 CandidateEvidence(깊은 버전)를 통합함 (2026-08-19)."""
    evidence_id: str
    topic: str
    handoff_section: str = ""
    text: str = ""  # 원문 인용용 - generate가 evidenceRefs.displayQuote로 그대로 인용
    structured_facts: dict[str, str] = {}
    importance_flags: list[str] = []
    requires_nurse_confirmation: bool = False


class TaskPriorityMeta(CamelModel):
    patient_status_urgency: str | None = None  # high 등 - 전체 enum 미확정
    time_constraint: str | None = None  # within_shift 등 - 전체 enum 미확정
    is_carry_over: bool = False


class OpenTask(CamelModel):
    """환자 업무(scopeType=PATIENT)와 공통/병동 업무(WARD|SUPPLY|ADMIN|ROOM|PERSONAL_SHIFT) 공용."""
    task_id: str
    title: str
    scope_type: str
    status: str
    patient_id: str | None = None
    required_before_handoff: bool = False
    priority_meta: TaskPriorityMeta | None = None


class VerifyDraftRequest(CamelModel):
    """노션 '인수인계 역검증'/'LLM 최종 제공 템플릿' 페이지 기준 (2026-08-19 재설계).
    이전엔 candidateSections(초안 생성 전 텍스트)를 검증했으나, 지금은 이미 생성된 초안
    (handoffDraft.items)을 검증하는 구조로 바뀜 - "초안 먼저 → 역검증 나중" 순서로 재확정됨."""
    request_id: str
    draft_id: str
    patient_id: str
    draft_items: list[DraftItemRef] = []
    candidate_evidence: list[Evidence] = []
    open_tasks: list[OpenTask] = []


class VerificationItem(CamelModel):
    """노션 예시 기준 카드 구조. severity는 HIGH/MEDIUM 예시뿐이라 전체 enum 미확정,
    type도 MISSING_HANDOFF_ITEM/OPEN_TASK_MISSING 예시뿐이라 전체 enum 미확정."""
    id: str
    patient_id: str
    topic: str
    type: str
    severity: str
    title: str
    reason: str
    suggested_question: str
    suggested_draft_text: str
    related_evidence_ids: list[str] = []
    related_task_ids: list[str] = []
    requires_nurse_confirmation: bool = True


class VerifyDraftResponse(CamelModel):
    request_id: str
    verification_items: list[VerificationItem] = []


# 인수인계 7개 섹션 (노션 "RAG 검색" 페이지 확정, PATIENT_STATUS/ACTIVITY는 기각된 안이라 안 씀)
HANDOFF_SECTIONS: dict[str, str] = {
    "VITAL_SIGNS": "활력징후",
    "RESPIRATION": "호흡",
    "MENTAL_STATUS": "의식상태",
    "PAIN": "통증",
    "TREATMENT": "처치",
    "DIET": "식이",
    "OBSERVATION": "관찰사항·특이사항",
}


class GenerateHandoffRequest(CamelModel):
    """2026-08-19: 공식 API 명세 DB의 SBAR/patients배치 구조 대신, 노션 "LLM 최종 제공 템플릿"
    1번 섹션(인수인계 최종 템플릿) 기준으로 재설계. 그 페이지의 "LLM이 받는 입력"에
    evidence(topic/handoffSection/structuredFacts/importanceFlags)와 "필요 시 환자 관련
    open task"가 명시돼 있어서 evidences를 Evidence로, openTasks를 추가함 (같은 날 재확인)."""
    request_id: str
    patient_id: str
    rounding_session_id: str
    evidences: list[Evidence] = []
    open_tasks: list[Task] = []


class EvidenceRef(CamelModel):
    evidence_id: str
    display_quote: str
    is_primary: bool = False


class HandoffDraftItem(CamelModel):
    topic: str
    section: str
    title: str
    summary: str
    requires_nurse_confirmation: bool = False
    confidence: float
    evidence_refs: list[EvidenceRef] = []


class GenerateHandoffResponse(CamelModel):
    draft_id: str
    patient_id: str
    rounding_session_id: str
    items: list[HandoffDraftItem] = []


