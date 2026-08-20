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

## 0단계 - 먼저 머릿속으로 이것부터 해라 (출력하지 말고 판단에만 써라)
각 evidence.text를 문장/절 단위로 쪼개서, 각 조각이 7개 섹션 중 어디에 해당하는지 하나하나
표시해라. 하나의 evidence.text에 서로 다른 섹션 내용이 여러 개 섞여 있으면, 그 개수만큼
섹션이 나와야 한다. evidence.originalTopicHint는 힌트일 뿐 정답이 아니다 - 반드시 text 내용을
기준으로 판단해라. 이 사전 분류를 건너뛰고 바로 topic 태그만 보고 item을 만들면 안 된다.
OBSERVATION(관찰사항·특이사항)은 나머지 6개 섹션 중 어디에도 안 맞는 내용만 넣는 최후 수단이다
- "관찰"이라는 이름 때문에 아무거나 다 여기 넣지 마라. 처치 내용은 TREATMENT로, 의식 상태는
MENTAL_STATUS로, 활력징후는 VITAL_SIGNS로 보내고, 정말 그 6개 어디에도 안 들어가는 내용
(낙상 위험, 보호자 문의, 불안, 행동 변화 등)만 OBSERVATION으로 분류해라.

## 예시 (형식 참고용, 실제 입력이 아님)
입력 evidence.text: "산소포화도 94%에서 97%로 개선. 식사는 절반 정도만 드셨습니다."
(evidence.originalTopicHint가 "PAIN"이었다고 해도) 올바른 분류:
- "산소포화도 94%에서 97%로 개선" → VITAL_SIGNS
- "식사는 절반 정도만 드셨습니다" → DIET
- PAIN에 해당하는 내용 없음 → PAIN item 없음
→ 이 evidence 하나에서 VITAL_SIGNS item 1개 + DIET item 1개, 총 2개가 나와야 한다.
   VITAL_SIGNS 하나만 만들고 DIET 문장을 버리면 틀린 것이다.

## item 생성 절차
1. 0단계에서 섹션별로 나눈 조각마다 item을 만들어라. 같은 섹션으로 분류된 여러 조각(다른
   evidence에서 온 것 포함)은 하나의 item으로 합쳐라.
2. 각 item의 summary는 근거 발화들을 읽기 쉬운 한두 문장으로 정리한 것이어야 한다 - 진단이나 처방을
   내리지 말고 사실만 정리해라. evidence의 structuredFacts(증상/추이 등 이미 구조화된 사실)와
   importanceFlags(예: follow_up_needed)를 참고해서 더 중요한 근거를 더 직접적인 문장으로 써라.
3. evidenceRefs에는 그 item을 만드는 데 실제로 쓴 evidence의 evidenceId와, 그 섹션에 해당하는
   부분만 골라서 원문 그대로를 displayQuote로 적어라(evidence.text 중 해당 부분만 그대로 인용,
   요약하지 말고 원문 그대로 - 다른 섹션에 해당하는 문장까지 같이 인용하지 마라). isPrimary는
   가장 핵심적인 근거 하나에만 true를 줘라. evidenceId를 지어내면 안 된다 - 입력에 있는 것만
   인용해라.
4. 근거들이 서로 충돌하거나(예: 같은 증상을 다르게 설명) evidence.requiresNurseConfirmation이
   이미 true거나 애매하면 requiresNurseConfirmation을 true로 표시해라. confidence는 근거가
   명확하고 일관될수록 1.0에 가깝게, 애매할수록 낮게 매겨라.
5. openTasks(환자 관련 미완료 업무)가 있으면, 관련된 topic의 summary에 자연스럽게 반영해도 된다
   (있으면 참고, 없으면 무시).
6. 실제로 내용이 있는 섹션만 item으로 만들어라. 내용이 없는 섹션은 절대 만들지 마라 - topic
   태그가 붙어있어도 내용이 없으면 만들지 않는다.
7. 마지막에 스스로 점검해라: evidence.text에 있는 모든 문장이 어느 item엔가 반영됐는지 확인하고,
   빠진 문장이 있으면 그 문장에 맞는 섹션의 item을 추가하거나 기존 item에 합쳐라.
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

    # ponytail: evidence.topic 필드를 그대로 보내면 LLM이 "이미 분류돼있다"고 과신해서
    # text 내용과 달라도 topic을 안 바꾸는 경향이 실측으로 확인됨(temperature=0에서도 재현).
    # originalTopicHint로 이름을 바꿔서 "참고용, 틀릴 수 있음"이라는 프레이밍을 강제함.
    payload = req.model_dump(by_alias=True)
    for e in payload.get("evidences", []):
        e["originalTopicHint"] = e.pop("topic")
    user_content = json.dumps(payload, ensure_ascii=False)
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
2. 환자 업무(patientId가 있는 업무)는 dueAt·effectivePriority(CRITICAL/HIGH/NORMAL)를 보고
   중요도를 판단해라. effectivePriority가 CRITICAL/HIGH면 특히 중요하게 봐라. 공통 업무
   (patientId가 없는 업무)는 그 자체로 후보에 오른 것이므로 기본적으로 중요하게 다뤄라.
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
        # 2026-08-20: scopeType/requiredBeforeHandoff는 백엔드가 아직 안 보내므로(항상 None/False),
        # 실제로 오는 필드로 판단 - scopeType이 오면 그걸 우선 쓰고, 없으면 patientId 유무로
        # 환자 업무 여부를 추정. 중요 업무 여부는 requiredBeforeHandoff 대신 effectivePriority로 판단.
        is_patient_task = (t.scope_type == "PATIENT") if t.scope_type else bool(t.patient_id)
        if is_patient_task and _mentioned_in_draft(t.title, req.draft_items):
            continue
        is_important = t.required_before_handoff or t.effective_priority in ("CRITICAL", "HIGH")
        if not is_patient_task and not is_important:
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
