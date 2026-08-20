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
| POST | `/internal/v1/handoffs/generate` | 근거(evidence)를 인수인계 7개 섹션으로 정리한 초안을 만든다 |
| POST | `/internal/v1/handoffs/precheck` | **이미 생성된 초안**을 근거·업무와 다시 대조해 누락·충돌·확인필요 카드를 만든다 (역검증) |

⚠️ 호출 순서 주의: `generate`가 먼저고 `precheck`(역검증)가 그 다음이다. 이름은 "precheck"(사전확인)이지만 실제로는 초안 생성 **후**에 그 초안을 검증하는 역할이다 — 2026-08-19 노션 "인수인계 역검증" 페이지 기준으로 순서가 이렇게 확정됐다. 엔드포인트 경로는 노션 공식 API 명세(`/internal/v1/handoffs/precheck`)를 그대로 유지했지만 내부 스키마는 완전히 새로 설계됐다.

세 개 다 `requestId`를 요청 그대로 응답에 에코하고(LLM이 베껴 쓰게 두지 않고 서버가 강제로 덮어씀), 성공 시 `201`을 반환한다. 입력이 비어 있으면(업무 0개, 환자 0명, 근거 0개) LLM을 호출하지 않고 즉시 빈 결과를 반환한다 — 불필요한 API 비용을 아끼기 위함.

**`/tasks/prioritize`**는 규칙 기반 1차 점수(`_rule_score`: 이월 여부 +3.0, 마감시각 있으면 +1.5, 위급 키워드 있으면 +2.0, 고위험 환자면 +2.0)를 먼저 계산한 뒤, gpt-4o-mini에게 그 점수의 근거를 한 문장으로 설명하게 한다. 점수 계산은 LLM한테 맡기지 않는다 — 같은 입력이면 항상 같은 점수가 나와야 하기 때문(재현성).

**`/handoffs/generate`**는 근거(`evidences`, `topic`/`handoffSection`/`structuredFacts`/`importanceFlags` 포함 - `precheck`의 `candidateEvidence`와 동일한 `Evidence` 스키마 공용)와, 참고용 미완료 업무(`openTasks`, 선택)를 받아서 인수인계 7개 섹션(topic) 기준으로 묶어 정리한다:

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

**`/handoffs/precheck`**(역검증)는 **이미 만들어진 초안**(`draftItems`)을 근거(`candidateEvidence`)·업무(`openTasks`)와 다시 대조한다. 먼저 규칙으로 후보를 좁힌다:
- 근거의 topic이 초안에 없으면 → `MISSING_HANDOFF_ITEM` 후보
- 근거가 `requiresNurseConfirmation=true`면 → `LOW_CONFIDENCE` 후보
- 환자 업무(`patientId`가 있는 업무)가 완료 안 됐는데 초안 요약에 제목의 의미있는 단어가 **전부** 들어있지 않으면 → `OPEN_TASK_MISSING` 후보 (2026-08-20: 단어 하나만 겹쳐도 "이미 언급됨"으로 스킵하던 걸 고침 — "혈압 재측정" 업무가 초안의 "혈압이 150/95로 상승" 관찰 기록과 `혈압`만 겹쳐서 오탐으로 사라지는 실사례 확인 후 수정)
- 공통 업무(`patientId` 없음)는 `effectivePriority`가 `CRITICAL`/`HIGH`면 후보 (2026-08-20: `scopeType`/`requiredBeforeHandoff`는 백엔드가 아직 안 보내는 걸로 확인돼 `patientId` 유무 + `effectivePriority`로 대체함 — 아래 "노션 명세와 다른 점" 참고)

이렇게 좁힌 후보만 LLM에 보내서, 진짜 문제인지 판단하고 카드(`verificationItems`)를 만든다. 각 카드는 `type`(`MISSING_HANDOFF_ITEM`/`OPEN_TASK_MISSING`/`CONFLICT`/`LOW_CONFIDENCE`), `severity`(`HIGH`/`MEDIUM`), 간호사에게 보여줄 `suggestedQuestion`, 초안에 바로 추가할 수 있는 `suggestedDraftText`를 포함한다.

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
- **역검증은 초안 생성 이후에 한다.** `precheck`라는 이름과 달리 "사전" 검증이 아니라, 이미 만들어진 초안을 다시 점검하는 단계다. 이 순서가 세션 내내 두 번 뒤집혔다(검증 먼저→초안 먼저→검증 나중) — 지금 것(초안 먼저, 역검증 나중)이 2026-08-19 새벽 "인수인계 역검증" 노션 페이지 기준 최종.
- 화자 임베딩은 현재 **MFCC mean/std 베이스라인**이다. 노션 설계 문서엔 SpeechBrain ECAPA로 적혀 있으나 아직 미적용 — 필요하면 `app/services/speaker_embedding.py`의 `SpeakerEmbeddingService`를 교체하면 된다. 실제 녹음 6개로 등록/재인식 테스트해보니 2/3만 정확히 맞았다 — 후보들 유사도가 0.98~0.99 사이에 몰려 있어 변별력이 약한 편(등록 품질이 나쁘면 바로 오매칭으로 이어짐).
- **`/handoffs/generate`는 `evidence.topic` 태그를 그대로 믿지 않는다.** LLM에 보낼 때 `topic` 필드명을 `originalTopicHint`로 바꿔서 보낸다 — "이미 분류된 정답"이라는 프레이밍을 주면 실제 텍스트 내용과 달라도 태그를 그대로 따라가는 경향이 실측으로 확인됐다(2026-08-20, 5개 시나리오 x 3회 재현). `temperature=0`도 같은 이유로 추가함(분류/판단 작업은 재현성이 창의성보다 중요).
- **pyannote 화자분리는 파일 경로 대신 waveform을 직접 넘긴다.** CPU 전용 이미지에서 `torchcodec`(오디오 디코딩용)이 CUDA 런타임(`libnvrtc`)을 요구해서 매번 조용히 실패하고 화자 1명으로 뭉개지는 문제가 있었다 — `soundfile`로 직접 읽어서 `{"waveform":..., "sample_rate":...}` 딕셔너리로 넘기는 방식으로 우회함(`app/services/diarization.py`).

## 노션 명세와 다른 점 (알려진 미확정 사항)

노션 API 명세에 예시가 하나뿐이라 우리가 추론해서 구현한 부분들. 백엔드와 재확인 필요:

- `severity`(`HIGH`/`MEDIUM`? precheck), `priority`, `PatientRisk.level`의 전체 enum 값
- `VerificationItem.type`의 전체 enum 값 (`MISSING_HANDOFF_ITEM`/`OPEN_TASK_MISSING` 외 `CONFLICT`/`LOW_CONFIDENCE`는 우리가 노션 설명 텍스트 보고 이름 붙인 것 — 노션에 명시적 enum 값으로 나온 건 아님)
- `X-Idempotency-Key`, `X-Request-Id` 헤더는 노션 명세상 필수지만 아직 서버에서 검증/활용하지 않음
- **precheck 요청의 정확한 shape** — 백엔드 확인 결과 "precheck AI 내부 입력 타입은 현재 `patients[].tasks[].id`로 되어 있다"는 답변을 받음. 우리 `openTasks`는 평평한 배열(`[{taskId, patientId, ...}]`)인데, 실제로 오는 게 이 모양이 맞는지, 아니면 `patients[].tasks[]`처럼 환자별로 중첩된 구조인지 아직 실제 요청 예시로 확인 못함 — 다음에 확인 필요

### 2026-08-20 백엔드 확인으로 해결된 것

- **`OpenTask` 실제 필드**: `scopeType`/`requiredBeforeHandoff`/`priorityMeta.patientStatusUrgency`는 노션 문서의 목표 스키마일 뿐 백엔드에 아직 구현 안 됨(안 옴). 실제로 안정적으로 오는 필드는 `taskId`, `patientId`, `title`, `description`, `dueAt`, `status`, `effectivePriority`(`CRITICAL`/`HIGH`/`NORMAL`, `confirmedPriority ?? rulePriority`). `scope_type`/`required_before_handoff`/`priority_meta`는 스키마에 optional로 남겨뒀다 — 나중에 백엔드가 구현하면 자동으로 쓰이도록.
- 이 변경으로 `precheck`의 환자/공통 업무 판단 로직을 `scopeType`/`requiredBeforeHandoff` 대신 `patientId` 유무 + `effectivePriority`로 다시 짬 (위 API 엔드포인트 섹션 참고). 이전 로직 그대로였으면 `OPEN_TASK_MISSING` 카드가 하나도 안 생기는 상태였음(백엔드가 안 보내는 필드에만 의존했었기 때문).

## Docker

이미지는 `jadest03/nurse-hand-ai:latest`로 Docker Hub에 public으로 올라가 있다 (CPU 전용, `python:3.11-slim` 기반, torch는 `--index-url https://download.pytorch.org/whl/cpu`로 설치해서 CUDA 패키지 없이 694MB 수준).

```bash
docker pull jadest03/nurse-hand-ai:latest
docker run -d --env-file .env -p 8000:8000 jadest03/nurse-hand-ai:latest
```

로컬에서 백엔드(`kimgt2828/nursehand-server`) + DB(postgres) + 이 서버까지 한 번에 띄우려면 `docker-compose.local.yml`을 쓴다 (경로는 로컬 절대경로 기준이라 각자 환경에 맞게 `.env`의 `AI_DATA_DIR`/`AI_TMP_DIR`/`POSTGRES_DATA_DIR`/`FILE_STORAGE_ROOT` 등을 바꿔야 함). **`docker-compose.prod.yml`은 이 저장소에 없다** — 실서버(가비아) 배포용 compose는 배포 담당자가 별도로 관리하는 것으로 보임, 확인 필요.

빌드 시 주의:
- **amd64로 빌드해야 한다.** 개발 환경(Apple Silicon)에서 기본 `docker build`를 쓰면 `arm64` 이미지가 나와서 실서버(리눅스 amd64)에서 못 돈다. `docker buildx build --platform linux/amd64 ...`로 빌드할 것.
- 이 네트워크에서는 `buildx ... --push` 직결이 자주 끊긴다(`broken pipe`) — `--load`로 로컬에 먼저 받은 다음 일반 `docker push`로 올리는 우회가 안정적이었음.
- `dev-dashboard.html`, `README.md`는 `.dockerignore`에 있어서 이미지에 안 들어간다 — 이 파일들만 고쳤으면 이미지 재빌드 불필요.

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
