from typing import Dict, Optional, Tuple

from fastapi import APIRouter, Depends

from app.auth import verify_internal_token
from app.llm import call_structured
from app.schemas import (
    ExtractTaskCandidate,
    ExtractTasksRequest,
    ExtractTasksResponse,
    PrioritizeTasksRequest,
    PrioritizeTasksResponse,
    PriorityResult,
    Task,
)

router = APIRouter(prefix="/internal/v1/tasks", tags=["tasks"], dependencies=[Depends(verify_internal_token)])

# ponytail: 키워드 기반 긴급도 휴리스틱. 실제 활력징후 추세를 반영하려면
# Task에 수치 데이터가 붙어야 함 - 나중에 업그레이드.
URGENT_KEYWORDS = ["산소", "통증", "출혈", "호흡", "의식", "혈압", "응급"]


def _rule_score(task: Task, is_high_risk: bool) -> tuple[float, int]:
    """규칙 점수와 매칭된 신호 개수를 같이 반환 - 신호 개수는 confidence 산정에 씀."""
    score = 0.0
    signal_count = 0
    if task.carried_over:  # ⑤ 이월 여부 - 가중치 최고
        score += 3.0
        signal_count += 1
    if task.due_at:  # ② 시간 제약
        score += 1.5
        signal_count += 1
    if any(k in task.title for k in URGENT_KEYWORDS):  # ① 환자 상태 긴급도(근사)
        score += 2.0
        signal_count += 1
    if is_high_risk:  # ③ 환자 위험도
        score += 2.0
        signal_count += 1
    return score, signal_count


def _priority_and_confidence(score: float, signal_count: int) -> tuple[str, str]:
    """2026-08-20: 백엔드 TaskAiSuggestionDto 기준(CRITICAL/HIGH/NORMAL + confidence)으로
    3단계 버킷 + 신호 개수 기반 confidence 산정. 임계값은 4개 신호(합 8.5)를 기준으로
    잡음 - 신호 3개 이상 겹치면 CRITICAL, 1개 이상이면 HIGH, 없으면 NORMAL."""
    if score >= 5.5:
        priority = "CRITICAL"
    elif score >= 2.0:
        priority = "HIGH"
    else:
        priority = "NORMAL"

    if signal_count >= 3:
        confidence = "HIGH"
    elif signal_count == 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    return priority, confidence


PRIORITY_REASON_SYSTEM_PROMPT = """너는 간호사가 직접 입력한 업무 목록에 우선순위를 매기는 도우미다.
각 업무는 아래 5가지 기준으로 판단한다:
1. 환자 상태 긴급도 - 상태 변화가 얼마나 심각한지(돌발 이벤트 여부)
2. 시간 제약 - 지금 안 하면 늦는 업무인지(투약 시간, 검사 예약 등)
3. 업무 성격 - 되돌릴 수 없는 일인지, 미뤄도 안전한 일인지(예: 투약 vs 퇴원교육)
4. 선행 업무 여부 - 이 업무를 먼저 해야 같은 환자의 다른 업무가 가능한지
   (같은 환자의 업무 제목들을 서로 비교해서 판단해라. 확실하지 않으면 선행관계로 보지 마라.)
5. 이월 여부 - 전 근무에서 넘어온 업무인지(이미 한 번 밀린 것이므로 가중치를 더 준다)

이미 규칙 기반으로 계산된 점수와 순서가 주어진다. 점수를 재계산하지 말고, 이미 주어진 점수·순서·
carriedOver·dueAt 값을 근거로 위 5가지 기준 중 어떤 게 이 업무에 해당하는지 한 문장으로 짧게 설명해라.
반드시 한국어로만 답해라."""

TASK_TITLE_RULES = [
    ("산소", "산소 상태 확인"),
    ("호흡", "호흡 상태 확인"),
    ("통증", "통증 재평가"),
    ("출혈", "출혈 상태 확인"),
    ("낙상", "낙상 후 상태 확인"),
    ("식사", "식이 섭취 확인"),
    ("섭취", "식이 섭취 확인"),
    ("배액", "배액 상태 확인"),
    ("드레싱", "드레싱 확인"),
]


@router.post("/extract", response_model=ExtractTasksResponse, status_code=201)
def extract_tasks(
    req: ExtractTasksRequest,
) -> ExtractTasksResponse:
    grouped: Dict[Tuple[Optional[str], str], dict] = {}

    for index, evidence in enumerate(req.evidence):
        title = _derive_task_title(evidence.summary)
        group_key = (evidence.patient_id, title)
        existing = grouped.get(group_key)

        if existing is None:
            grouped[group_key] = {
                "candidate": ExtractTaskCandidate(
                    candidate_key=f"candidate-{index + 1}",
                    patient_id=evidence.patient_id,
                    title=title,
                    description=evidence.summary.strip(),
                    due_at=None,
                    evidence_source_ids=[evidence.source_id],
                ),
            }
            continue

        candidate: ExtractTaskCandidate = existing["candidate"]
        if evidence.source_id not in candidate.evidence_source_ids:
            candidate.evidence_source_ids.append(evidence.source_id)

    return ExtractTasksResponse(
        request_id=req.request_id,
        candidates=[item["candidate"] for item in grouped.values()],
    )


@router.post("/prioritize", response_model=PrioritizeTasksResponse, status_code=201)
def prioritize_tasks(
    req: PrioritizeTasksRequest,
) -> PrioritizeTasksResponse:
    risk_by_patient = {r.patient_id: r.level for r in req.patient_risk}

    def _to_result(t: Task) -> PriorityResult:
        is_high_risk = risk_by_patient.get(t.patient_id) == "HIGH"
        score, signal_count = _rule_score(t, is_high_risk)
        priority, confidence = _priority_and_confidence(score, signal_count)
        return PriorityResult(
            task_id=t.task_id,
            score=score,
            priority=priority,
            confidence=confidence,
            reasons=["[stub] 규칙 기준 자동 산정"],
        )

    scored = sorted((_to_result(t) for t in req.tasks), key=lambda r: r.score, reverse=True)
    stub = PrioritizeTasksResponse(request_id=req.request_id, results=scored)
    if not req.tasks:
        return stub

    user_content = stub.model_dump_json(by_alias=True)
    result = call_structured(PRIORITY_REASON_SYSTEM_PROMPT, user_content, PrioritizeTasksResponse, stub)
    result.request_id = req.request_id  # LLM이 베껴 쓰게 두지 않고 항상 원본으로 덮어씀

    # 2026-08-20: score/priority/confidence는 재현성이 생명이라 LLM이 절대 못 바꾸게 강제 덮어씀.
    # LLM이 taskId를 빠뜨리거나 순서를 바꿔도 우리 규칙 계산 순서(stub)를 최종 기준으로 삼는다.
    stub_by_id = {r.task_id: r for r in scored}
    llm_reasons_by_id = {r.task_id: r.reasons for r in result.results if r.task_id in stub_by_id}
    result.results = [
        PriorityResult(
            task_id=r.task_id,
            score=r.score,
            priority=r.priority,
            confidence=r.confidence,
            reasons=llm_reasons_by_id.get(r.task_id, r.reasons),
        )
        for r in scored
    ]
    return result


def _derive_task_title(summary: str) -> str:
    normalized = summary.strip()
    for keyword, title in TASK_TITLE_RULES:
        if keyword in normalized:
            return title
    return "라운딩 후속 확인"
