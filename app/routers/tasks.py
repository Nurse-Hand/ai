from fastapi import APIRouter, Depends

from app.auth import verify_internal_token
from app.errors import INTERNAL_ERROR_RESPONSES, InferenceFailure, InferenceFailureCode
from app.llm import call_structured
from app.task_contracts import (
    ExtractTasksRequest,
    ExtractTasksResponse,
    PrioritizeTasksRequest,
    PrioritizeTasksResponse,
)

router = APIRouter(
    prefix="/internal/v1/tasks",
    tags=["internal-tasks"],
    dependencies=[Depends(verify_internal_token)],
    responses=INTERNAL_ERROR_RESPONSES,
)

EXTRACT_SYSTEM_PROMPT = """Extract follow-up task candidates only from the supplied evidence.
Never invent a patient or evidence ID. Do not decide task state, schedule, final priority, or ordering.
Return the exact response schema."""

PRIORITIZE_SYSTEM_PROMPT = """Suggest a priority, reasons, and confidence for every candidate.
Priority must be CRITICAL, HIGH, or NORMAL. Never emit a numeric score, LOW priority, final decision, or ordering.
Copy candidateKey and evidenceSourceIds only from the input and return the exact response schema."""


def _invalid_response() -> None:
    raise InferenceFailure(InferenceFailureCode.INVALID_RESPONSE)


@router.post("/extract", response_model=ExtractTasksResponse, status_code=200)
def extract_tasks(req: ExtractTasksRequest) -> ExtractTasksResponse:
    result = call_structured(
        EXTRACT_SYSTEM_PROMPT, req.model_dump_json(by_alias=True), ExtractTasksResponse
    )
    if result.request_id != req.request_id:
        _invalid_response()
    evidence_by_id = {item.source_id: item for item in req.evidence}
    seen: set[str] = set()
    for candidate in result.candidates:
        if candidate.candidate_key in seen:
            _invalid_response()
        seen.add(candidate.candidate_key)
        referenced = [evidence_by_id.get(source_id) for source_id in candidate.evidence_source_ids]
        if any(item is None for item in referenced):
            _invalid_response()
        if candidate.patient_id is not None and not any(
            item is not None and item.patient_id == candidate.patient_id for item in referenced
        ):
            _invalid_response()
    return result


@router.post("/prioritize", response_model=PrioritizeTasksResponse, status_code=200)
def prioritize_tasks(req: PrioritizeTasksRequest) -> PrioritizeTasksResponse:
    result = call_structured(
        PRIORITIZE_SYSTEM_PROMPT, req.model_dump_json(by_alias=True), PrioritizeTasksResponse
    )
    if result.request_id != req.request_id or len(result.suggestions) != len(req.candidates):
        _invalid_response()
    candidates = {item.candidate_key: item for item in req.candidates}
    seen: set[str] = set()
    for suggestion in result.suggestions:
        candidate = candidates.get(suggestion.candidate_key)
        if candidate is None or suggestion.candidate_key in seen:
            _invalid_response()
        seen.add(suggestion.candidate_key)
        if not set(suggestion.evidence_source_ids).issubset(set(candidate.evidence_source_ids)):
            _invalid_response()
    return result
