# Nurse Hand AI Server

Nurse Hand(간호사 인수인계 보조 앱)의 AI 서버다. 두 영역을 함께 다룬다.

1. **음성 분석** — STT + 화자 분리 + 화자 매칭 (`/api/*`, 인증 불필요, Mobile → 이 서버)
2. **AI 판단 로직** — 업무 우선순위 산정, 인수인계 누락 검증(역질문), 인수인계 초안 생성 (`/internal/v1/*`, `X-Internal-Token` 인증 필요, Node.js 백엔드 → 이 서버)

두 영역 모두 하나의 FastAPI 앱(`main.py`)에서 서비스한다.

## 빠른 시작

```bash
python3.10 -m venv .venv   # 3.10 권장 - 이유는 아래 "왜 python3.10인가" 참고
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # 값 채우기 (아래 "환경 변수" 참고)
uvicorn main:app --port 8000 --reload
```

실제 화자 분리(pyannote)를 쓰려면 추가로 설치한다:

```bash
pip install -r requirements-ai.txt
```

`requirements-ai.txt`가 없으면(`.env`에 `PYANNOTE_AUTH_TOKEN`을 넣어도) pyannote 모듈 자체가 없어서 화자 분리가 안 되고, 전체 오디오를 화자 1명(`SPEAKER_00`)으로 처리하는 fallback으로만 동작한다.

### 왜 python3.10인가

pyannote.audio 4.x가 요구하는 `torch`/`torchaudio` 조합(`torch==2.11.0`, `torchaudio==2.11.0`)이 macOS에서 python 3.13/3.14용 wheel을 아직 안 내놓는다. python3.10에는 wheel이 있다. Docker 배포판(`python:3.11-slim` 기준)에서는 이 문제가 없을 수 있으니, 이미지 빌드 시 실제로 설치되는지 재확인할 것.

## 환경 변수

`.env.example`을 복사해서 채운다.

| 변수 | 설명 |
|---|---|
| `OPENAI_API_KEY` | 업무/인수인계 AI 판단용. 없으면 해당 엔드포인트가 stub(가짜 응답)을 반환한다 — 500 에러가 아니라 조용히 가짜 데이터가 나가니 배포 전 꼭 확인할 것 |
| `OPENAI_MODEL` | 기본 `gpt-4o-mini` |
| `INTERNAL_API_TOKEN` | 백엔드 → 이 서버 호출 인증 토큰 (`X-Internal-Token` 헤더 값과 대조) |
| `DEEPGRAM_API_KEY` | STT. 없으면 로컬 STT(`LOCAL_STT_MODEL_DIR`, sherpa-onnx) 또는 빈 transcript로 fallback |
| `DEEPGRAM_MODEL`, `DEEPGRAM_LANGUAGE` | 기본 `nova-3`, `ko-KR` |
| `PYANNOTE_AUTH_TOKEN` | HuggingFace 토큰. `pyannote/speaker-diarization-community-1` 모델 라이선스 동의 필요 |
| `PYANNOTE_DIARIZATION_MODEL` | 기본값 그대로 두면 됨 |
| `MATCH_THRESHOLD` | 화자 매칭 유사도 임계값 (기본 0.7) - 이 이상이어야 `bestMatch`로 확정 |
| `TOP_K_DEFAULT` | 화자 후보 개수 (기본 3) |
| `MIN_SEGMENT_SEC`, `MIN_SPEAKER_TOTAL_SEC` | 너무 짧은 구간/화자 필터링 기준 |
| `DATA_DIR`, `TMP_DIR` | 화자 DB(`speakers.json`)와 업로드 임시파일 저장 경로 |

## API 엔드포인트

### 업무/인수인계 AI 판단 (`/internal/v1/*`, 인증 필요)

| 메서드 | 경로 | 하는 일 |
|---|---|---|
| POST | `/internal/v1/tasks/prioritize` | 간호사가 직접 입력한 업무 목록에 우선순위 점수를 매긴다 |
| POST | `/internal/v1/handoffs/precheck` | 최근 기록과 인수인계 후보 내용을 대조해 빠진 부분을 역질문으로 만든다 |
| POST | `/internal/v1/handoffs/generate` | 근거(evidence)를 인수인계 7개 섹션으로 정리한 초안을 만든다 |

세 개 다 `requestId`를 요청 그대로 응답에 에코하고(LLM이 베껴 쓰게 두지 않고 서버가 강제로 덮어씀), 성공 시 `201`을 반환한다. 입력이 비어 있으면(업무 0개, 환자 0명, 근거 0개) LLM을 호출하지 않고 즉시 빈 결과를 반환한다 — 불필요한 API 비용을 아끼기 위함.

**`/tasks/prioritize`**는 규칙 기반 1차 점수(`_rule_score`: 이월 여부 +3.0, 마감시각 있으면 +1.5, 위급 키워드 있으면 +2.0, 고위험 환자면 +2.0)를 먼저 계산한 뒤, gpt-4o-mini에게 그 점수의 근거를 한 문장으로 설명하게 한다. 점수 계산은 LLM한테 맡기지 않는다 — 같은 입력이면 항상 같은 점수가 나와야 하기 때문(재현성).

**`/handoffs/precheck`**는 미완료 업무를 먼저 텍스트 겹침으로 1차 필터링해서(`_mentioned_in_sections`) 이미 후보 섹션에 언급된 건 LLM에 안 보낸다. 그 다음 최근 기록(`recentEvents`)을 후보 섹션(`candidateSections`)과 대조해서, 없는 내용만 확인형 질문으로 만든다.

**`/handoffs/generate`**는 근거를 인수인계 7개 섹션(topic) 기준으로 묶어서 정리한다:

| topic | 화면 표시 |
|---|---|
| `VITAL_SIGNS` | 활력징후 |
| `RESPIRATION` | 호흡 |
| `MENTAL_STATUS` | 의식상태 |
| `PAIN` | 통증 |
| `TREATMENT` | 처치 |
| `DIET` | 식이 |
| `OBSERVATION` | 관찰사항·특이사항 |

같은 topic의 여러 근거는 하나의 항목으로 압축되고, 각 항목은 실제로 인용한 근거(`evidenceRefs`, 원문 그대로 인용)를 달고 나온다. 근거가 서로 충돌하거나 애매하면 `requiresNurseConfirmation: true`로 표시된다.

### 음성 분석 (`/api/*`, 인증 불필요)

| 메서드 | 경로 | 하는 일 |
|---|---|---|
| GET | `/health` | 서버 상태 + Deepgram/pyannote 설정 여부 확인 |
| POST | `/api/diarization/analyze` | 오디오 업로드 → STT + 화자 분리 + 등록 화자 매칭 |
| GET | `/api/speakers` | 등록된 화자(voice profile) 목록 조회 |
| DELETE | `/api/speakers/{speaker_id}` | 잘못 등록된 화자 삭제 |
| POST | `/api/speakers/register-from-diarization` | 분석 결과에서 특정 화자 구간만 이어붙여 새 화자로 등록 |

`/api/diarization/analyze` 처리 순서: 업로드 → ffmpeg로 16kHz mono WAV 정규화 → Deepgram STT + pyannote 화자 분리(병렬) → 화자별 구간을 이어붙여 임베딩 추출(현재 MFCC mean/std 기반) → 등록된 화자 DB와 cosine similarity 비교 → Top-K 후보 + threshold 이상이면 `bestMatch` 반환. **화자를 자동 확정하지 않는다** — 후보만 제시하고 간호사가 확인한다.

## 테스트

```bash
pytest                          # 스키마/인증/기본 흐름 (실제 LLM 호출 없이, stub 경로)
python e2e_test.py              # 서버를 띄운 채로 실행 - 실제 OpenAI 호출로 3개 엔드포인트 전체 흐름 확인
```

`e2e_test.py`는 `uvicorn main:app --port 8000`이 떠 있어야 하고, `.env`의 `INTERNAL_API_TOKEN`을 그대로 사용한다.

## 아키텍처 결정 배경 (헷갈리기 쉬운 것들)

- **업무는 AI가 추출하지 않는다.** 간호사가 직접 입력해서 DB에 저장하고, 우리는 `/tasks/prioritize`로 우선순위만 매긴다. 예전엔 `/tasks/extract`(AI 자동 추출)가 있었으나 삭제됨.
- **원본 음성/발화는 이 서버가 직접 다루지 않는다.** `/api/diarization/analyze`가 STT+화자분리까지만 하고, 그 결과를 evidence로 가공·저장하는 건 Node.js 백엔드 몫이다.
- **인수인계 템플릿은 SBAR가 아니다.** 위 7개 topic 기반 구조로 2026-08-19에 재설계됨.
- 화자 임베딩은 현재 **MFCC mean/std 베이스라인**이다. 노션 설계 문서엔 SpeechBrain ECAPA로 적혀 있으나 아직 미적용 — 필요하면 `app/services/speaker_embedding.py`의 `SpeakerEmbeddingService`를 교체하면 된다.

## 노션 명세와 다른 점 (알려진 미확정 사항)

노션 API 명세에 예시가 하나뿐이라 우리가 추론해서 구현한 부분들. 백엔드와 재확인 필요:

- `PrecheckRequest.candidateSections`의 정확한 내부 형태 (현재 `dict[str, str]`로 가정)
- `severity`(`CRITICAL`/`RECOMMENDED`?), `priority`, `PatientRisk.level`의 전체 enum 값
- `ClinicalEvent.type`의 전체 enum 값 (`OBSERVATION` 외 후보: 와이어프레임 기준 관찰/일반/중요/상담/보고/처치 추정)
- `GenerateHandoffRequest`의 `evidences` 구조는 "LLM 최종 제공 템플릿" 노션 페이지의 예시를 기반으로 역설계한 것 — 백엔드가 실제로 이 모양 그대로 보낼지 미확정
- `X-Idempotency-Key`, `X-Request-Id` 헤더는 노션 명세상 필수지만 아직 서버에서 검증/활용하지 않음

## 디렉토리 구조

```
main.py                    # 앱 진입점, 라우터 등록
app/
  config.py                 # 환경변수 (pydantic-settings)
  deps.py                   # 화자분리 관련 서비스 싱글턴 (lru_cache)
  auth.py                   # X-Internal-Token 인증
  llm.py                    # OpenAI structured output 호출 (call_structured)
  schemas.py                # 전체 Pydantic 스키마
  routers/
    tasks.py                 # /internal/v1/tasks/prioritize
    handoffs.py               # /internal/v1/handoffs/{precheck,generate}
    diarization.py            # /api/diarization/analyze
    speakers.py                # /api/speakers/*
  services/
    diarization.py             # pyannote 파이프라인
    transcription.py            # Deepgram/로컬 STT
    speaker_embedding.py         # 화자 임베딩 추출/유사도
    speaker_store.py              # 화자 DB (JSON 파일)
    audio.py                       # 업로드/정규화/오디오 유틸
    analysis.py                     # 분석 결과 조합
tests/                      # pytest (stub 경로 검증)
e2e_test.py                # 수동 통합 테스트 (실제 LLM 호출)
```
