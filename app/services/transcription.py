from pathlib import Path
from typing import Optional

import httpx

from app.config import Settings
from app.schemas import ServerTranscript, ServerTranscriptUtterance


class TranscriptionService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def transcribe(
        self,
        wav_path: Path,
    ) -> tuple[ServerTranscript, list[ServerTranscriptUtterance]]:
        if self.settings.deepgram_api_key:
            return await self._transcribe_deepgram(wav_path)

        if self.settings.local_stt_model_dir:
            result = self._transcribe_local(wav_path)
            if result is not None:
                return result

        transcript = ServerTranscript(
            provider="none",
            model="unconfigured",
            language=self.settings.deepgram_language,
            text="",
            confidence=None,
        )
        return transcript, []

    async def _transcribe_deepgram(
        self,
        wav_path: Path,
    ) -> tuple[ServerTranscript, list[ServerTranscriptUtterance]]:
        params = {
            "model": self.settings.deepgram_model,
            "language": self.settings.deepgram_language,
            "smart_format": "true",
            "punctuate": "true",
            "paragraphs": "true",
            "utterances": "true",
        }
        headers = {
            "Authorization": f"Token {self.settings.deepgram_api_key}",
            "Content-Type": "audio/wav",
        }
        audio = wav_path.read_bytes()
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                params=params,
                headers=headers,
                content=audio,
            )
            response.raise_for_status()
            payload = response.json()

        alternative = (
            payload.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
        )
        transcript = ServerTranscript(
            provider="deepgram",
            model=self.settings.deepgram_model,
            language=self.settings.deepgram_language,
            text=alternative.get("transcript", "") or "",
            confidence=alternative.get("confidence"),
        )
        utterances_payload = payload.get("results", {}).get("utterances", []) or []
        utterances = [
            ServerTranscriptUtterance(
                utteranceId=f"utt-{index + 1}",
                startSec=round(float(item.get("start", 0.0)), 3),
                endSec=round(float(item.get("end", 0.0)), 3),
                durationSec=round(float(item.get("end", 0.0)) - float(item.get("start", 0.0)), 3),
                transcript=item.get("transcript", "") or "",
                confidence=item.get("confidence"),
                deepgramSpeaker=(
                    None
                    if item.get("speaker") is None
                    else f"SPEAKER_{int(item['speaker']):02d}"
                ),
                bestMatch=None,
                candidates=[],
            )
            for index, item in enumerate(utterances_payload)
        ]

        if not utterances and transcript.text:
            utterances = [
                ServerTranscriptUtterance(
                    utteranceId="utt-1",
                    startSec=0.0,
                    endSec=0.0,
                    durationSec=0.0,
                    transcript=transcript.text,
                    confidence=transcript.confidence,
                    deepgramSpeaker=None,
                    bestMatch=None,
                    candidates=[],
                )
            ]

        return transcript, utterances

    def _transcribe_local(
        self,
        wav_path: Path,
    ) -> Optional[tuple[ServerTranscript, list[ServerTranscriptUtterance]]]:
        try:
            import sherpa_onnx  # type: ignore
            import soundfile as sf
        except Exception:
            return None

        if self.settings.local_stt_model_dir is None:
            return None

        model_dir = self.settings.local_stt_model_dir
        tokens = model_dir / "tokens.txt"
        encoder = model_dir / "encoder-epoch-99-avg-1.int8.onnx"
        decoder = model_dir / "decoder-epoch-99-avg-1.int8.onnx"
        joiner = model_dir / "joiner-epoch-99-avg-1.int8.onnx"
        if not all(path.exists() for path in [tokens, encoder, decoder, joiner]):
            return None

        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            tokens=str(tokens),
            num_threads=2,
            decoding_method="greedy_search",
        )
        audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        mono = audio[:, 0]
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, mono)
        recognizer.decode_stream(stream)
        text = stream.result.text
        transcript = ServerTranscript(
            provider="local_zipformer_korean",
            model="sherpa-onnx-zipformer-korean",
            language="ko-KR",
            text=text,
            confidence=None,
        )
        utterances = [
            ServerTranscriptUtterance(
                utteranceId="utt-1",
                startSec=0.0,
                endSec=round(len(mono) / sample_rate, 3),
                durationSec=round(len(mono) / sample_rate, 3),
                transcript=text,
                confidence=None,
                deepgramSpeaker=None,
                bestMatch=None,
                candidates=[],
            )
        ]
        return transcript, utterances
