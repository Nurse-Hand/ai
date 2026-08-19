from fastapi import APIRouter, Depends

from app.auth import verify_internal_token
from app.llm import call_structured
from app.schemas import (
    PrioritizeTasksRequest,
    PrioritizeTasksResponse,
    PriorityResult,
    Task,
)

router = APIRouter(prefix="/internal/v1/tasks", tags=["tasks"], dependencies=[Depends(verify_internal_token)])

# ponytail: 키워드 기반 긴급도 휴리스틱. 실제 활력징후 추세를 반영하려면
# Task에 수치 데이터가 붙어야 함 - 나중에 업그레이드.
URGENT_KEYWORDS = ["산소", "통증", "출혈", "호흡", "의식", "혈압", "응급"]


def _rule_score(task: Task) -> float:
    score = 0.0
    if task.carried_over:  # ⑤ 이월 여부 - 가중치 최고
        score += 3.0
    if task.due_at:  # ② 시간 제약
        score += 1.5
    if any(k in task.title for k in URGENT_KEYWORDS):  # ① 환자 상태 긴급도(근사)
        score += 2.0
    return score


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


@router.post("/prioritize", response_model=PrioritizeTasksResponse, status_code=201)
def prioritize_tasks(
    req: PrioritizeTasksRequest,
) -> PrioritizeTasksResponse:
    risk_by_patient = {r.patient_id: r.level for r in req.patient_risk}

    def score_with_risk(task: Task) -> float:
        score = _rule_score(task)
        if risk_by_patient.get(task.patient_id) == "HIGH":  # ③ 환자 위험도
            score += 2.0
        return score

    def _to_result(t: Task) -> PriorityResult:
        score = score_with_risk(t)
        return PriorityResult(
            task_id=t.task_id,
            score=score,
            priority="CRITICAL" if score >= 4.0 else "LOW",
            reasons=["[stub] 규칙 기준 자동 산정"],
        )

    scored = sorted((_to_result(t) for t in req.tasks), key=lambda r: r.score, reverse=True)
    stub = PrioritizeTasksResponse(request_id=req.request_id, results=scored)
    if not req.tasks:
        return stub

    user_content = stub.model_dump_json(by_alias=True)
    result = call_structured(PRIORITY_REASON_SYSTEM_PROMPT, user_content, PrioritizeTasksResponse, stub)
    result.request_id = req.request_id  # LLM이 베껴 쓰게 두지 않고 항상 원본으로 덮어씀
    return result
