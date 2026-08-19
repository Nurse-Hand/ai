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


class ClinicalEvent(CamelModel):
    """노션 확정 스키마 (`/internal/v1/handoffs/precheck`의 recentEvents).
    2026-08-19: 업무는 간호사 직접 입력으로 결정돼 `/tasks/extract`는 삭제됨 - 이제 precheck 전용.
    누가 TimelineNote를 이 형태로 변환하는지는 아직 노션에 명시 안 됨 - 우리 AI 서버는
    이 형태를 계약으로 받는다는 것만 확정."""
    event_id: str
    patient_id: str
    type: str  # 노션 예시는 "OBSERVATION" 하나뿐, 전체 enum 미확정
    summary: str


class Task(CamelModel):
    """노션 확정 스키마 (`/internal/v1/tasks/prioritize`의 tasks, precheck의 pendingTasks 공용)."""
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


class PrecheckItem(CamelModel):
    """노션 확정 스키마 (`/internal/v1/handoffs/precheck` 응답 items).
    답변 4버튼(NO_ISSUE|INCLUDE_HANDOFF|UNVERIFIED|NOT_APPLICABLE)은 우리 응답에 없음 -
    프론트가 고정 표시, 백엔드의 별도 '역질문 응답 저장' API가 answer를 받는 구조."""
    patient_id: str
    severity: str  # 노션 예시 "CRITICAL" 하나뿐, summary의 critical/recommended로 미루어 최소 2종 추정, 미확정
    question: str
    evidence_event_ids: list[str] = []
    reason: str


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


class PatientPrecheckInput(CamelModel):
    patient_id: str
    recent_events: list[ClinicalEvent] = []
    candidate_sections: dict[str, str] = {}  # 노션 예시는 {} 뿐 - field_id: 텍스트로 추정, 미확정
    pending_tasks: list[Task] = []


class PrecheckRequest(CamelModel):
    request_id: str
    patients: list[PatientPrecheckInput]
    lookback_shifts: int = 3


class PrecheckResponse(CamelModel):
    request_id: str
    items: list[PrecheckItem]


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


class DraftEvidence(CamelModel):
    """노션 '파트별 사항-AI > LLM 최종 제공 템플릿' 기준 generate 입력 근거 단위.
    evidenceRefs(응답)의 evidenceId/displayQuote와 짝을 이룸."""
    evidence_id: str
    topic: str  # HANDOFF_SECTIONS 키 중 하나
    text: str


class GenerateHandoffRequest(CamelModel):
    """2026-08-19: 공식 API 명세 DB의 SBAR/patients배치 구조 대신, 노션 '파트별 사항-AI'
    3개 페이지(AI 음성 분석 기능/VAD와 화자 분리/LLM 최종 제공 템플릿) 기준으로 재설계."""
    request_id: str
    patient_id: str
    rounding_session_id: str
    evidences: list[DraftEvidence] = []


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


