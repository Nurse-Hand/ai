from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.schemas import SpeakerRecord


class SpeakerStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / "speakers.json"

    def list(self) -> List[SpeakerRecord]:
        if not self.file_path.exists():
            return []
        payload = json.loads(self.file_path.read_text(encoding="utf-8"))
        return [SpeakerRecord.model_validate(item) for item in payload.get("speakers", [])]

    def save_all(self, speakers: List[SpeakerRecord]) -> None:
        payload = {
            "speakers": [speaker.model_dump(mode="json") for speaker in speakers],
        }
        self.file_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def upsert(
        self,
        speaker_id: str,
        display_name: str,
        embedding: list[float],
        sample_path: Optional[str] = None,
    ) -> SpeakerRecord:
        now = datetime.now(timezone.utc)
        speakers = self.list()
        existing = next((item for item in speakers if item.speakerId == speaker_id), None)
        if existing is None:
            record = SpeakerRecord(
                speakerId=speaker_id,
                displayName=display_name,
                registeredAt=now,
                updatedAt=now,
                embedding=embedding,
                samplePath=sample_path,
            )
            speakers.append(record)
        else:
            record = existing.model_copy(
                update={
                    "displayName": display_name,
                    "updatedAt": now,
                    "embedding": embedding,
                    "samplePath": sample_path or existing.samplePath,
                }
            )
            speakers = [
                record if item.speakerId == speaker_id else item
                for item in speakers
            ]
        self.save_all(speakers)
        return record

    def delete(self, speaker_id: str) -> bool:
        speakers = self.list()
        filtered = [item for item in speakers if item.speakerId != speaker_id]
        if len(filtered) == len(speakers):
            return False
        self.save_all(filtered)
        return True
