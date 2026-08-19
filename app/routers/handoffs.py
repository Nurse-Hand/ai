from fastapi import APIRouter, Depends, HTTPException

from app.auth import verify_internal_token
from app.errors import INTERNAL_ERROR_RESPONSES, InferenceFailure, InferenceFailureCode
from app.handoff_contracts import (
    EvidenceReference,
    GenerateHandoffRequest,
    GenerateHandoffResponse,
    HandoffPrecheckRequest,
    HandoffPrecheckResponse,
)
from app.llm import call_structured

router = APIRouter(
    prefix="/internal/v1/handoffs",
    tags=["internal-handoffs"],
    dependencies=[Depends(verify_internal_token)],
    responses=INTERNAL_ERROR_RESPONSES,
)

SECTIONS = {"PATIENT_STATUS", "PAIN", "TREATMENT", "DIET", "ACTIVITY", "OBSERVATION"}

PRECHECK_PROMPT = """Review the frozen patient snapshot before draft generation.
Return only necessary confirmation questions with CRITICAL or RECOMMENDED severity.
Every evidence reference must identify an input timeline event or task for the same patient."""
GENERATE_PROMPT = """Generate NURSING_HANDOFF_V1 from the frozen snapshots and stored precheck answers.
Return all six sections for every input patient and preserve source citations.
Never promote unverified information to fact. When includeUnverified is true, mark every UNVERIFIED item with its required warning.
Do not finalize a handoff or store any state."""


def _invalid_response() -> None:
    raise InferenceFailure(InferenceFailureCode.INVALID_RESPONSE)


def _input_error(message: str) -> None:
    raise HTTPException(status_code=422, detail={"code": "INVALID_INPUT", "message": message})


def _registry(patients):
    patient_ids = set()
    sources = {}
    for patient in patients:
        if patient.patient_id in patient_ids:
            _input_error("patientId must be unique")
        patient_ids.add(patient.patient_id)
        for event in patient.timeline_events:
            key = ("TIMELINE_EVENT", event.id)
            if key in sources:
                _input_error("snapshot source IDs must be unique")
            sources[key] = patient.patient_id
        for task in patient.tasks:
            key = ("TASK", task.id)
            if key in sources:
                _input_error("snapshot source IDs must be unique")
            sources[key] = patient.patient_id
    return patient_ids, sources


def _evidence_key(reference: EvidenceReference):
    return (reference.source_type, reference.source_id, reference.patient_id)


def _validate_evidence(references, sources, patient_id, allow_empty=False):
    if not references and not allow_empty:
        _invalid_response()
    seen = set()
    for reference in references:
        key = _evidence_key(reference)
        if key in seen:
            _invalid_response()
        seen.add(key)
        if reference.patient_id != patient_id:
            _invalid_response()
        if sources.get((reference.source_type, reference.source_id)) != patient_id:
            _invalid_response()


@router.post("/precheck", response_model=HandoffPrecheckResponse, status_code=200)
def precheck_handoff(req: HandoffPrecheckRequest) -> HandoffPrecheckResponse:
    patient_ids, sources = _registry(req.patients)
    result = call_structured(
        PRECHECK_PROMPT, req.model_dump_json(by_alias=True), HandoffPrecheckResponse
    )
    if result.request_id != req.request_id:
        _invalid_response()
    question_keys = set()
    for question in result.questions:
        if question.question_key in question_keys or question.patient_id not in patient_ids:
            _invalid_response()
        question_keys.add(question.question_key)
        _validate_evidence(question.evidence, sources, question.patient_id)
    return result


@router.post("/generate", response_model=GenerateHandoffResponse, status_code=200)
def generate_handoff(req: GenerateHandoffRequest) -> GenerateHandoffResponse:
    patient_ids, sources = _registry(req.patients)
    item_ids = set()
    unverified = {}
    for item in req.precheck_items:
        if item.id in item_ids:
            _input_error("precheck item IDs must be unique")
        item_ids.add(item.id)
        evidence_patients = {reference.patient_id for reference in item.evidence}
        if len(evidence_patients) != 1:
            _input_error("precheck evidence must belong to one patient")
        patient_id = next(iter(evidence_patients))
        for reference in item.evidence:
            if sources.get((reference.source_type, reference.source_id)) != patient_id:
                _input_error("precheck evidence is not in the frozen snapshot")
        if item.answer == "UNVERIFIED":
            unverified[item.id] = (patient_id, {_evidence_key(value) for value in item.evidence})

    result = call_structured(
        GENERATE_PROMPT, req.model_dump_json(by_alias=True), GenerateHandoffResponse
    )
    if result.request_id != req.request_id or len(result.patients) != len(patient_ids):
        _invalid_response()
    seen_patients = set()
    for patient in result.patients:
        if patient.patient_id not in patient_ids or patient.patient_id in seen_patients:
            _invalid_response()
        seen_patients.add(patient.patient_id)
        if {section.section for section in patient.sections} != SECTIONS:
            _invalid_response()
        for section in patient.sections:
            _validate_evidence(section.citations, sources, patient.patient_id, allow_empty=True)
    if seen_patients != patient_ids:
        _invalid_response()

    if not req.include_unverified and result.warnings:
        _invalid_response()
    seen_warnings = set()
    for warning in result.warnings:
        expected = unverified.get(warning.item_id)
        if expected is None or warning.item_id in seen_warnings:
            _invalid_response()
        seen_warnings.add(warning.item_id)
        expected_patient, expected_evidence = expected
        if warning.patient_id != expected_patient:
            _invalid_response()
        warning_evidence = [_evidence_key(value) for value in warning.evidence]
        if len(warning_evidence) != len(set(warning_evidence)):
            _invalid_response()
        if set(warning_evidence) != expected_evidence:
            _invalid_response()
    if req.include_unverified and seen_warnings != set(unverified):
        _invalid_response()
    return result
