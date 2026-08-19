import json
import uuid

from fastapi import APIRouter, Depends

from app.auth import verify_internal_token
from app.llm import call_structured
from app.schemas import (
    HANDOFF_SECTIONS,
    GenerateHandoffRequest,
    GenerateHandoffResponse,
    PrecheckRequest,
    PrecheckResponse,
    PrecheckItem,
)

router = APIRouter(prefix="/internal/v1/handoffs", tags=["handoffs"], dependencies=[Depends(verify_internal_token)])

GENERATE_SYSTEM_PROMPT = f"""너는 근거(evidence) 목록을 인수인계 초안 항목으로 정리하는 도우미다.
인수인계는 아래 7개 섹션(topic)만 쓴다: {", ".join(f'{k}({v})' for k, v in HANDOFF_SECTIONS.items())}

절차:
1. evidences를 topic별로 묶어라. 같은 topic의 여러 근거는 하나의 item으로 압축해라.
2. 각 item의 summary는 근거 발화들을 읽기 쉬운 한두 문장으로 정리한 것이어야 한다 - 진단이나 처방을
   내리지 말고 사실만 정리해라.
3. evidenceRefs에는 그 item을 만드는 데 실제로 쓴 evidence의 evidenceId와 원문 그대로를
   displayQuote로 적어라. isPrimary는 가장 핵심적인 근거 하나에만 true를 줘라. evidenceId를
   지어내면 안 된다 - 입력에 있는 것만 인용해라.
4. 근거들이 서로 충돌하거나(예: 같은 증상을 다르게 설명) 애매하면 requiresNurseConfirmation을
   true로 표시해라. confidence는 근거가 명확하고 일관될수록 1.0에 가깝게, 애매할수록 낮게 매겨라.
5. 근거가 없는 topic은 item을 만들지 마라. 억지로 채우지 마라.
반드시 한국어로만 답해라."""


@router.post("/generate", response_model=GenerateHandoffResponse, status_code=201)
def generate_handoff(req: GenerateHandoffRequest) -> GenerateHandoffResponse:
    draft_id = f"handoff-draft-{uuid.uuid4()}"
    stub = GenerateHandoffResponse(
        draft_id=draft_id,
        patient_id=req.patient_id,
        rounding_session_id=req.rounding_session_id,
        items=[],
    )
    if not req.evidences:
        return stub

    user_content = req.model_dump_json(by_alias=True)
    result = call_structured(GENERATE_SYSTEM_PROMPT, user_content, GenerateHandoffResponse, stub)
    # LLM이 베껴 쓰게 두지 않고 식별자는 항상 원본/서버 생성값으로 덮어씀
    result.draft_id = draft_id
    result.patient_id = req.patient_id
    result.rounding_session_id = req.rounding_session_id
    return result


PRECHECK_SYSTEM_PROMPT = """너는 인수인계 후보 섹션(candidateSections)을 최근 임상 이벤트(recentEvents)·
미완료 업무와 대조해 빠진 부분만 확인형 질문으로 만드는 도우미다. 환자 여러 명이 배치로 주어진다.

절차를 반드시 이 순서로 따라라:
1. 각 환자의 recentEvents 항목마다 핵심 임상 사실을 하나씩 나열한다.
2. 나열한 사실마다, 같은 환자의 candidateSections 값 전체를 글자 그대로 다시 읽고
   "이 사실과 같은 내용이 이미 candidateSections 안에 있는가"를 확인한다.
   동의어나 다른 표현으로라도 이미 있으면 "있음"으로 처리한다.
3. "있음"으로 확인된 사실은 절대 질문으로 만들지 않는다.
4. candidateSections 어디에도 없는 사실만 질문으로 만든다. evidenceEventIds에는 그 사실의 근거가 된
   eventId를 정확히 적어라 - 입력에 없는 eventId를 지어내면 안 된다.
5. unmentionedPendingTasks로 주어진 미완료 업무는 이미 규칙으로 "candidateSections에 언급 안 됨"이
   확인된 것들이다 - 이 목록에 있는 업무는 전부 "이 업무가 처리/확인됐는지" 묻는 질문으로 만들어라.
6. severity는 환자 상태에 중대한 영향이면 "CRITICAL", 그 외 확인이 권장되는 수준이면 "RECOMMENDED"로 판단해라.

이름 표기, 문장 형식, 어투 같은 사소하거나 행정적인 차이는 절대 지적하지 마라.
확실하지 않으면 질문을 만들지 말고 넘어가라 - 애매할 땐 침묵이 낫다.
질문 문장은 확인형으로만 써라 ("~하셨나요?"). 지시형 문장("~하셔야 합니다")은 금지한다.
진단이나 처방을 제안하지 마라.
반드시 한국어로만 답해라."""


def _mentioned_in_sections(task_title: str, candidate_sections: dict[str, str]) -> bool:
    # ponytail: 단어 겹침 기반의 단순 체크. 동의어/다른 표현은 못 잡음 - 필요시 LLM 기반으로 업그레이드.
    combined = " ".join(candidate_sections.values())
    words = [w for w in task_title.split() if len(w) >= 2]
    return bool(words) and any(w in combined for w in words)


@router.post("/precheck", response_model=PrecheckResponse, status_code=201)
def precheck_handoff(req: PrecheckRequest) -> PrecheckResponse:
    payload_patients = []
    stub_items = []
    for p in req.patients:
        # 미완료 업무는 먼저 규칙(텍스트 겹침)으로 1차 필터링 - 이미 언급된 건 LLM한테 안 보냄
        unmentioned = [t for t in p.pending_tasks if not _mentioned_in_sections(t.title, p.candidate_sections)]
        payload_patients.append(
            {
                "patientId": p.patient_id,
                "recentEvents": [e.model_dump(by_alias=True) for e in p.recent_events],
                "candidateSections": p.candidate_sections,
                "unmentionedPendingTasks": [t.model_dump(by_alias=True) for t in unmentioned],
            }
        )
        if p.recent_events or unmentioned:
            stub_items.append(
                PrecheckItem(
                    patient_id=p.patient_id,
                    severity="RECOMMENDED",
                    question="[stub] 최근 기록 대비 확인이 필요한 항목이 있나요?",
                    reason="[stub] 근거 없음 (OPENAI_API_KEY 미설정)",
                )
            )

    stub = PrecheckResponse(request_id=req.request_id, items=stub_items)
    if not req.patients:
        return stub

    payload = {
        "requestId": req.request_id,
        "lookbackShifts": req.lookback_shifts,
        "patients": payload_patients,
    }
    user_content = json.dumps(payload, ensure_ascii=False)
    result = call_structured(PRECHECK_SYSTEM_PROMPT, user_content, PrecheckResponse, stub)
    result.request_id = req.request_id  # LLM이 베껴 쓰게 두지 않고 항상 원본으로 덮어씀
    return result
