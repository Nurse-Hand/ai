from collections import defaultdict
from typing import Optional

from app.config import Settings
from app.schemas import (
    DiarizationCandidate,
    DiarizationSegment,
    DiarizationSpeakerMatch,
    RoundingDiarizationAnalysis,
    ServerTranscript,
    ServerTranscriptUtterance,
    SpeakerRecord,
)
from app.services.diarization import RawSegment
from app.services.speaker_embedding import SpeakerEmbeddingService


def find_best_overlap(
    utterance: ServerTranscriptUtterance,
    segments: list[RawSegment],
) -> tuple[Optional[str], float]:
    best_speaker = None
    best_overlap = 0.0
    for segment in segments:
        overlap = max(
            0.0,
            min(utterance.endSec, segment.end_sec) - max(utterance.startSec, segment.start_sec),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment.diarized_speaker
    return best_speaker, round(best_overlap, 3)


def top_candidates(
    embedding_service: SpeakerEmbeddingService,
    probe_embedding: Optional[list[float]],
    registered_speakers: list[SpeakerRecord],
    top_k: int,
    threshold: float,
) -> tuple[Optional[DiarizationCandidate], list[DiarizationCandidate]]:
    if not probe_embedding:
        return None, []
    candidates = sorted(
        [
            DiarizationCandidate(
                speakerId=speaker.speakerId,
                displayName=speaker.displayName,
                similarity=round(
                    embedding_service.cosine_similarity(probe_embedding, speaker.embedding),
                    3,
                ),
            )
            for speaker in registered_speakers
        ],
        key=lambda item: item.similarity,
        reverse=True,
    )[:top_k]
    best = candidates[0] if candidates and candidates[0].similarity >= threshold else None
    return best, candidates


def build_analysis_response(
    *,
    file_name: str,
    diarization_available: bool,
    transcript: ServerTranscript,
    utterances: list[ServerTranscriptUtterance],
    raw_segments: list[RawSegment],
    speaker_embeddings: dict[str, Optional[list[float]]],
    registered_speakers: list[SpeakerRecord],
    embedding_service: SpeakerEmbeddingService,
    settings: Settings,
    top_k: int,
) -> RoundingDiarizationAnalysis:
    by_speaker: dict[str, list[RawSegment]] = defaultdict(list)
    for segment in raw_segments:
        by_speaker[segment.diarized_speaker].append(segment)

    speaker_candidates: dict[str, tuple[Optional[DiarizationCandidate], list[DiarizationCandidate]]] = {}
    for diarized_speaker, embedding in speaker_embeddings.items():
        speaker_candidates[diarized_speaker] = top_candidates(
            embedding_service,
            embedding,
            registered_speakers,
            top_k,
            settings.match_threshold,
        )

    segments = [
        DiarizationSegment(
            startSec=segment.start_sec,
            endSec=segment.end_sec,
            durationSec=segment.duration_sec,
            diarizedSpeaker=segment.diarized_speaker,
            bestMatch=speaker_candidates.get(segment.diarized_speaker, (None, []))[0],
            candidates=speaker_candidates.get(segment.diarized_speaker, (None, []))[1],
        )
        for segment in raw_segments
    ]

    annotated_utterances: list[ServerTranscriptUtterance] = []
    for utterance in utterances:
        diarized_speaker, overlap = find_best_overlap(utterance, raw_segments)
        best_match, candidates = speaker_candidates.get(diarized_speaker or "", (None, []))
        annotated_utterances.append(
            utterance.model_copy(
                update={
                    "diarizedSpeaker": diarized_speaker,
                    "overlapSeconds": overlap,
                    "bestMatch": best_match,
                    "candidates": candidates,
                }
            )
        )

    speaker_matches: list[DiarizationSpeakerMatch] = []
    for diarized_speaker, speaker_segments in by_speaker.items():
        total_speech_sec = round(sum(segment.duration_sec for segment in speaker_segments), 3)
        relevant_utterances = [
            utterance
            for utterance in annotated_utterances
            if utterance.diarizedSpeaker == diarized_speaker and utterance.transcript.strip()
        ]
        representative_quote = None
        if relevant_utterances:
            representative_quote = max(
                relevant_utterances,
                key=lambda item: item.durationSec,
            )
        best_match, candidates = speaker_candidates.get(diarized_speaker, (None, []))
        speaker_matches.append(
            DiarizationSpeakerMatch(
                diarizedSpeaker=diarized_speaker,
                segmentCount=len(speaker_segments),
                totalSpeechSec=total_speech_sec,
                embeddingAvailable=speaker_embeddings.get(diarized_speaker) is not None,
                bestMatch=best_match,
                candidates=candidates,
                representativeQuote=representative_quote,
            )
        )

    speaker_matches.sort(key=lambda item: item.totalSpeechSec, reverse=True)

    return RoundingDiarizationAnalysis(
        fileName=file_name,
        diarizationAvailable=diarization_available,
        threshold=settings.match_threshold,
        rawSegmentCount=len(raw_segments),
        minSegmentSec=settings.min_segment_sec,
        minSpeakerTotalSec=settings.min_speaker_total_sec,
        maxSpeakerTotalSec=settings.max_speaker_total_sec,
        totalSegments=len(segments),
        segments=segments,
        speakerMatches=speaker_matches,
        transcript=transcript,
        utterances=annotated_utterances,
    )
