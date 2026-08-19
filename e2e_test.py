"""수동 e2e 스모크 테스트.

2026-08-19 기준 확정된 흐름만 검증한다:
  - /internal/v1/audio/analyze, /internal/v1/tasks/extract는 삭제됨
    (STT/화자분리는 backend-ai가, 업무 입력은 간호사 직접 입력이 담당)
  - 이 서버가 실제로 맡는 건 tasks/prioritize + handoffs/precheck + handoffs/generate 뿐

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

print("\n=== STEP B: 오늘 기록 vs 후보 초안 대조 (역질문) ===")
precheck_result = post(
    "/internal/v1/handoffs/precheck",
    {
        "requestId": str(uuid.uuid4()),
        "patients": [
            {
                "patientId": "301",
                "recentEvents": [
                    {
                        "eventId": "e1",
                        "patientId": "301",
                        "type": "OBSERVATION",
                        "summary": "최근 3일간 SpO2 94%~97% 반복 변동",
                    }
                ],
                "candidateSections": {"situation": "환자 상태 안정적"},
                "pendingTasks": [
                    {"taskId": "t1", "patientId": "301", "title": "산소포화도 재측정", "carriedOver": True}
                ],
            }
        ],
        "lookbackShifts": 3,
    },
)
for item in precheck_result["items"]:
    print(f"  [{item['severity']}] {item['question']} - 근거: {item['reason']}")

print("\n=== STEP C: 인수인계 초안 생성 (evidence 기반, 7섹션 topic) ===")
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

print("\n=== 끝까지 정상 완료 ===")
