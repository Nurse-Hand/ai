"""로컬 내부 API 계약 스모크 검사.

실제 OpenAI, Deepgram, HuggingFace 호출을 만들지 않는다. 실행 중인 서버를 대상으로 한
외부 통합 검증은 Node application Port가 확정된 뒤 별도 환경에서 수행한다.
"""

from main import app

INTERNAL_PATHS = {
    "/internal/v1/tasks/extract",
    "/internal/v1/tasks/prioritize",
    "/internal/v1/handoffs/precheck",
    "/internal/v1/handoffs/generate",
    "/internal/v1/audio/analyze",
}


def main() -> int:
    schema = app.openapi()
    for path in sorted(INTERNAL_PATHS):
        operation = schema["paths"][path]["post"]
        if "200" not in operation["responses"] or "201" in operation["responses"]:
            raise RuntimeError(f"unexpected success contract: {path}")
        if operation.get("security") != [{"APIKeyHeader": []}]:
            raise RuntimeError(f"missing internal authentication: {path}")
    print("Internal AI contract smoke passed: 5 authenticated 200 endpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
