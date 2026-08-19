# Nurse Hand AI Server

Node.js 서버가 호출하는 동기 AI 추론 서비스다. 내부 계약의 단일 기준은
[`openapi/internal-ai-v1.json`](openapi/internal-ai-v1.json)이며, 다섯 endpoint는 모두
`X-Internal-Token` 인증과 성공 상태 `200 OK`를 사용한다.

## 책임 경계

Python은 입력 snapshot을 바탕으로 후보와 초안만 제안한다. Job과 업무 상태, 최종 업무
우선순위, 정렬, schedule, 인수인계 최종 확정, 환자·화자 확정, 장기 원본 파일 및 화자
profile 저장은 Node.js가 소유한다. 요청 원문, transcript 전문, token은 로그에 남기지 않는다.

OpenAI key가 없거나 STT/diarization이 구성되지 않은 경우 가짜 정상 응답을 만들지 않고
`AI_UPSTREAM_UNAVAILABLE`로 실패한다. deterministic fake는 pytest에서만 주입한다.

## 내부 API

| Method | Path | Result |
|---|---|---|
| POST | `/internal/v1/tasks/extract` | evidence 기반 업무 후보와 `HIGH | MEDIUM | LOW` confidence |
| POST | `/internal/v1/tasks/prioritize` | `CRITICAL | HIGH | NORMAL` 제안, reasons, confidence |
| POST | `/internal/v1/handoffs/precheck` | 초안 생성 전 `CRITICAL | RECOMMENDED` 확인 질문 |
| POST | `/internal/v1/handoffs/generate` | frozen snapshot과 저장된 답변 기반 6-section 초안 |
| POST | `/internal/v1/audio/analyze` | 동기 STT, utterance, diarized speaker candidate envelope |

Task evidence ID와 Handoff citation은 요청 snapshot에 포함된 ID만 응답할 수 있다.
`prioritize`는 숫자 score, `LOW` priority, 최종 확정값 또는 정렬 결과를 반환하지 않는다.

Handoff 호출 순서는 `precheck → 답변 저장/동결 snapshot → generate`다. template은
`NURSING_HANDOFF_V1`이고 section은 아래 여섯 개를 정확히 한 번씩 반환한다.

- `PATIENT_STATUS`
- `PAIN`
- `TREATMENT`
- `DIET`
- `ACTIVITY`
- `OBSERVATION`

`UNVERIFIED` 답변을 포함하도록 요청한 경우에도 사실로 승격하지 않으며
`UNVERIFIED_INFORMATION` warning과 동일한 source evidence를 유지한다.

Audio는 `multipart/form-data`의 `audio`와 `sourceAudioFileId`를 받는다. 응답 utterance는
`speakerLabel`, `startedAtMs`, `endedAtMs`, `text`, `confidence`,
`sourceAudioFileId`를 가진다. candidate schema는 화자별 최대 3개와 similarity를 지원하지만,
Node Audio Port와 profile 입력 계약이 없는 현재 Adapter에서는 후보 조회가 비활성이고 빈 배열을
반환할 수 있다. Python JSON profile 저장을 이 경로에 연결하지 않으며 자동 확정 필드도 없다.
처리가 끝나면 내부 임시 파일을 검증하며 삭제하고, 삭제 실패 시 성공을 반환하지 않는다.

## 오류

| Status | Code |
|---|---|
| 401 | `INVALID_INTERNAL_TOKEN` |
| 422 | `INVALID_INPUT` |
| 429 | `AI_UPSTREAM_RATE_LIMITED` |
| 502 | `AI_UPSTREAM_INVALID_RESPONSE` |
| 503 | `AI_UPSTREAM_UNAVAILABLE` |
| 504 | `AI_UPSTREAM_TIMEOUT` |

오류 message에는 token, 입력 원문, transcript, upstream 응답 전문을 넣지 않는다.

## 환경 변수

| Variable | Description |
|---|---|
| `INTERNAL_TOKEN` | `X-Internal-Token`과 비교할 서비스 token |
| `OPENAI_API_KEY` | Task/Handoff structured inference |
| `OPENAI_MODEL` | 기본 `gpt-4o-mini` |
| `AI_TIMEOUT_SECONDS` | OpenAI timeout, 기본 30초 |
| `DEEPGRAM_API_KEY` | STT provider |
| `PYANNOTE_AUTH_TOKEN` | diarization model token |
| `TMP_DIR` | 요청 처리 중에만 사용하는 임시 파일 경로 |
| `AUDIO_MAX_UPLOAD_BYTES` | audio file byte 상한, 기본 25 MiB |
| `AUDIO_MAX_REQUEST_BYTES` | multipart request byte 상한, 기본 26 MiB |
| `AUDIO_PROCESSING_TIMEOUT_SECONDS` | ffmpeg/STT/diarization timeout, 기본 120초 |

## 로컬 실행과 검증

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/uvicorn main:app --port 8000
.venv/Scripts/pytest
python scripts/generate_openapi.py --check
python -m scripts.generate_openapi --check
```

OpenAPI를 의도적으로 변경한 경우 다음 명령으로 artifact를 다시 생성한다.

```bash
python scripts/generate_openapi.py
```

pytest fake는 실제 OpenAI, Deepgram, HuggingFace 호출이나 실제 환자·음성 fixture를 사용하지 않는다.

## 기존 외부 route 상태

기존 `/api/diarization/**`와 `/api/speakers/**`는 호환성 결정 전까지 삭제하지 않았다.
새 `/internal/v1/**` route와 인증·소유권을 분리했다. 기존 외부 route는 JSON 화자 저장 기능을
포함하므로 Node 연동 대상으로 간주하지 않으며, 제거 또는 호환 정책은 별도 결정이 필요하다.

## 확인된 server 연동 gap

- server Task extract Port의 exact-key validator에는 아직 candidate `confidence`가 없다.
  Python OpenAPI를 채택하려면 Port와 Adapter에 같은 enum을 추가해야 한다.
- server `origin/dev`에는 audio analyze application Port가 아직 없다. 현재 Python 입력은
  Issue #16/#17과 기존 analyze route에서 확인되는 최소 `audio + sourceAudioFileId`만 선언한다.
  환자 segment와 Node 소유 화자 profile 후보를 전달하는 입력 필드는 Port 확정 전까지 추가하지
  않았다. 따라서 profile 입력이 확정되기 전 내부 route의 candidate 배열은 비어 있을 수 있다.
