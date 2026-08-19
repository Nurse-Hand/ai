import json
import uuid

from fastapi import APIRouter, Depends

from app.auth import verify_internal_token
from app.llm import call_structured
from app.schemas import (
    HANDOFF_SECTIONS,
    GenerateHandoffRequest,
    GenerateHandoffResponse,
    VerificationItem,
    VerifyDraftRequest,
    VerifyDraftResponse,
)

router = APIRouter(prefix="/internal/v1/handoffs", tags=["handoffs"], dependencies=[Depends(verify_internal_token)])

GENERATE_SYSTEM_PROMPT = f"""너는 근거(evidence) 목록을 인수인계 초안 항목으로 정리하는 도우미다.
인수인계는 아래 7개 섹션(topic)만 쓴다: {", ".join(f'{k}({v})' for k, v in HANDOFF_SECTIONS.items())}

절차:
1. evidences를 topic별로 묶어라. 같은 topic의 여러 근거는 하나의 item으로 압축해라.
2. 각 item의 summary는 근거 발화들을 읽기 쉬운 한두 문장으로 정리한 것이어야 한다 - 진단이나 처방을
   내리지 말고 사실만 정리해라. evidence의 structuredFacts(증상/추이 등 이미 구조화된 사실)와
   importanceFlags(예: follow_up_needed)를 참고해서 더 중요한 근거를 더 직접적인 문장으로 써라.
3. evidenceRefs에는 그 item을 만드는 데 실제로 쓴 evidence의 evidenceId와 원문 그대로를
   displayQuote로 적어라(evidence.text 필드를 그대로 인용). isPrimary는 가장 핵심적인 근거
   하나에만 true를 줘라. evidenceId를 지어내면 안 된다 - 입력에 있는 것만 인용해라.
4. 근거들이 서로 충돌하거나(예: 같은 증상을 다르게 설명) evidence.requiresNurseConfirmation이
   이미 true거나 애매하면 requiresNurseConfirmation을 true로 표시해라. confidence는 근거가
   명확하고 일관될수록 1.0에 가깝게, 애매할수록 낮게 매겨라.
5. openTasks(환자 관련 미완료 업무)가 있으면, 관련된 topic의 summary에 자연스럽게 반영해도 된다
   (있으면 참고, 없으면 무시).
6. 근거가 없는 topic은 item을 만들지 마라. 억지로 채우지 마라.
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


PRECHECK_SYSTEM_PROMPT = """너는 이미 생성된 인수인계 초안(draftItems)을, 근거(candidateEvidence)와
미완료 업무(openTasks)를 기준으로 다시 점검하는 역검증 도우미다. 초안을 새로 쓰지 않는다.

절차:
1. candidateSeeds로 주어진 항목들은 이미 규칙 기반으로 "초안에 없을 가능성이 있다"고 걸러진 후보다.
   각 후보가 진짜로 다음 근무자에게 전달해야 할 내용인지, 아니면 무시해도 되는 잡음인지 판단해라.
2. 환자 업무(scopeType=PATIENT)는 patientStatusUrgency·timeConstraint·isCarryOver를 보고 중요도를
   판단해라. 공통 업무(scopeType != PATIENT)는 requiredBeforeHandoff/isCarryOver가 true면 특히
   중요하게 봐라.
3. 같은 topic의 candidateEvidence 중 structuredFacts가 서로 다르게 설명하는 것이 있으면(예: 같은
   증상을 환자와 보호자가 다르게 말함) type을 "CONFLICT"로, reason에 어느 근거끼리 충돌하는지 적어라.
4. type은 MISSING_HANDOFF_ITEM(초안에 없는 근거) | OPEN_TASK_MISSING(초안에 없는 업무) |
   CONFLICT(근거 충돌) | LOW_CONFIDENCE(근거는 있으나 불확실) 중 하나로 판단해라.
5. severity는 환자 상태·안전에 직접 영향이면 HIGH, 그 외 확인이 권장되는 수준이면 MEDIUM으로 판단해라.
6. suggestedQuestion은 간호사가 한 번에 읽고 답할 수 있는 짧은 확인형 질문으로 써라
   ("~을 인계에 포함할까요?"). 지시형 문장은 금지한다.
7. suggestedDraftText는 초안에 바로 추가할 수 있는 한 줄 인수인계 문장으로 써라 - 진단·처방 금지,
   사실만 정리.
8. relatedEvidenceIds/relatedTaskIds에는 입력에 있는 evidenceId/taskId만 적어라 - 지어내면 안 된다.
9. 후보가 진짜 문제없다고 판단되면 카드를 만들지 마라 - 애매할 땐 만들지 않는 게 낫다.
반드시 한국어로만 답해라."""


def _topic_in_draft(topic: str, draft_items: list) -> bool:
    return any(item.topic == topic for item in draft_items)


def _mentioned_in_draft(task_title: str, draft_items: list) -> bool:
    # ponytail: 단어 겹침 기반의 단순 체크. 동의어/다른 표현은 못 잡음 - 필요시 LLM 기반으로 업그레이드.
    combined = " ".join(item.summary for item in draft_items)
    words = [w for w in task_title.split() if len(w) >= 2]
    return bool(words) and any(w in combined for w in words)


def _rule_based_candidates(req: VerifyDraftRequest) -> list[dict]:
    """규칙 기반 1차 검증 - LLM 부르기 전에 후보만 좁힌다 (노션 '인수인계 역검증' 페이지 기준)."""
    candidates = []
    for e in req.candidate_evidence:
        if not _topic_in_draft(e.topic, req.draft_items):
            candidates.append({"seedType": "MISSING_HANDOFF_ITEM", "evidence": e.model_dump(by_alias=True)})
        elif e.requires_nurse_confirmation:
            candidates.append({"seedType": "LOW_CONFIDENCE", "evidence": e.model_dump(by_alias=True)})

    for t in req.open_tasks:
        if t.status == "DONE":
            continue
        is_patient_task = t.scope_type == "PATIENT"
        if is_patient_task and _mentioned_in_draft(t.title, req.draft_items):
            continue
        if not is_patient_task and not t.required_before_handoff:
            continue
        candidates.append({"seedType": "OPEN_TASK_MISSING", "task": t.model_dump(by_alias=True)})

    return candidates


@router.post("/precheck", response_model=VerifyDraftResponse, status_code=201)
def precheck_handoff(req: VerifyDraftRequest) -> VerifyDraftResponse:
    candidates = _rule_based_candidates(req)
    stub = VerifyDraftResponse(
        request_id=req.request_id,
        verification_items=[
            VerificationItem(
                id=f"verify-stub-{i}",
                patient_id=req.patient_id,
                topic=c.get("evidence", c.get("task", {})).get("topic", "TASK"),
                type=c["seedType"],
                severity="MEDIUM",
                title="[stub] 확인 필요 항목",
                reason="[stub] 근거 없음 (OPENAI_API_KEY 미설정)",
                suggested_question="[stub] 이 내용을 인계에 포함할까요?",
                suggested_draft_text="",
            )
            for i, c in enumerate(candidates)
        ],
    )
    if not candidates:
        return VerifyDraftResponse(request_id=req.request_id, verification_items=[])

    payload = {
        "requestId": req.request_id,
        "draftId": req.draft_id,
        "patientId": req.patient_id,
        "draftItems": [d.model_dump(by_alias=True) for d in req.draft_items],
        "candidateEvidence": [e.model_dump(by_alias=True) for e in req.candidate_evidence],
        "openTasks": [t.model_dump(by_alias=True) for t in req.open_tasks],
        "candidateSeeds": candidates,
    }
    user_content = json.dumps(payload, ensure_ascii=False)
    result = call_structured(PRECHECK_SYSTEM_PROMPT, user_content, VerifyDraftResponse, stub)
    result.request_id = req.request_id  # LLM이 베껴 쓰게 두지 않고 항상 원본으로 덮어씀
    return result
