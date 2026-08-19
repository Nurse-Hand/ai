"""수동 e2e 스모크 테스트.

2026-08-19 밤 기준 확정된 흐름 (순서 중요 - "초안 먼저 → 역검증 나중"):
  1. tasks/prioritize - 간호사가 직접 입력한 업무 우선순위 산정
  2. handoffs/generate - 근거(evidence)로 인수인계 초안 생성 (7섹션 topic 구조)
  3. handoffs/precheck - 방금 생성된 초안을 역검증 (누락/충돌/확인필요 카드 생성)
     "인수인계 역검증" 노션 페이지 기준으로 재설계됨 - candidateSections 대조 방식 아님

실행 전: `uvicorn main:app --port 8000`으로 서버 띄워두고,
INTERNAL_API_TOKEN 환경변수를 .env와 동일하게 맞춰서 실행.
`requests`는 앱 자체엔 필요 없어서 requirements.txt에서 뺐음 - 이 스크립트만 쓰려면 별도 설치 필요.
"""

import os
import uuid

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "http://localhost:8000"
TOKEN = os.environ["INTERNAL_API_TOKEN"]  # .env에서 로드 - 하드코딩 금지
HEADERS = {"X-Internal-Token": TOKEN, "Content-Type": "application/json"}


def post(path, body):
    resp = requests.post(f"{BASE}{path}", headers=HEADERS, json=body)
    resp.raise_for_status()
    return resp.json()


print("=== STEP A: 간호사가 직접 입력한 업무 우선순위 산정 ===")
tasks = [
    {"taskId": "t1", "patientId": "301", "title": "산소포화도 재측정", "carriedOver": True},
    {"taskId": "t2", "patientId": "405", "title": "통증 재평가", "dueAt": "2026-08-19T10:00:00+09:00"},
    {"taskId": "t3", "patientId": "212", "title": "CT 결과 확인 후 의사 보고"},
]
patient_risk = [{"patientId": "301", "level": "HIGH"}]
prioritize_result = post(
    "/internal/v1/tasks/prioritize",
    {
        "requestId": str(uuid.uuid4()),
        "tasks": tasks,
        "patientRisk": patient_risk,
        "now": "2026-08-19T09:00:00+09:00",
    },
)
for r in prioritize_result["results"]:
    print(f"  [{r['priority']}] score={r['score']} taskId={r['taskId']} - {r['reasons']}")

print("\n=== STEP B: 인수인계 초안 생성 (evidence 기반, 7섹션 topic) ===")
generate_result = post(
    "/internal/v1/handoffs/generate",
    {
        "requestId": str(uuid.uuid4()),
        "patientId": "301",
        "roundingSessionId": "round-e2e-test",
        "evidences": [
            {
                "evidenceId": "ev-1",
                "topic": "VITAL_SIGNS",
                "text": "301호 김OO 환자분, 산소포화도 94%에서 97%로 개선, 비강캐뉼라 산소 2L 유지",
            }
        ],
    },
)
for item in generate_result["items"]:
    print(f"  [{item['topic']}/{item['section']}] {item['title']}: {item['summary']} (confidence={item['confidence']})")

print("\n=== STEP C: 방금 생성된 초안을 역검증 (누락/충돌/확인필요 카드) ===")
# 초안엔 VITAL_SIGNS만 있는데, 근거는 RESPIRATION도 있다고 가정 -> 누락 카드가 나와야 함
draft_items = [{"topic": item["topic"], "summary": item["summary"]} for item in generate_result["items"]]
precheck_result = post(
    "/internal/v1/handoffs/precheck",
    {
        "requestId": str(uuid.uuid4()),
        "draftId": generate_result["draftId"],
        "patientId": "301",
        "draftItems": draft_items,
        "candidateEvidence": [
            {
                "evidenceId": "ev-2",
                "topic": "RESPIRATION",
                "handoffSection": "호흡",
                "structuredFacts": {"symptom": "기침", "trend": "야간 악화"},
                "importanceFlags": ["follow_up_needed"],
                "requiresNurseConfirmation": False,
            }
        ],
        "openTasks": [
            {
                "taskId": "t1",
                "title": "산소포화도 재측정",
                "scopeType": "PATIENT",
                "status": "TODO",
                "patientId": "301",
                "priorityMeta": {"patientStatusUrgency": "high", "isCarryOver": True},
            }
        ],
    },
)
for item in precheck_result["verificationItems"]:
    print(f"  [{item['severity']}/{item['type']}] {item['title']}")
    print(f"     이유: {item['reason']}")
    print(f"     제안 질문: {item['suggestedQuestion']}")
    print(f"     제안 문장: {item['suggestedDraftText']}")

print("\n=== 끝까지 정상 완료 ===")
