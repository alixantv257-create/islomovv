from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import requests

from client_interface import SpeechClient


EMOTION_TAGS = {
    "gazab": "[angry] [shouting]",
    "qaygu": "[sad]",
    "yiglash": "[sobbing] [sad]",
    "qorquv": "[scared] [panting]",
    "hayrat": "[surprised]",
    "xursandchilik": "[happy]",
    "kulish": "[laughing]",
    "sovuqqonlik": "[calm]",
    "qahramonona": "[confident] [determined]",
    "yovuz_qahramon": "[disdainful] [confident]",
}


@dataclass(frozen=True)
class FishTTSConfig:
    api_key: str
    base_url: str = "https://api.fish.audio"
    model: str = "s2-pro"
    default_reference_id: str = ""
    sample_rate: int = 24000
    format: str = "wav"
    latency: str = "normal"
    temperature: float = 0.55
    top_p: float = 0.7
    chunk_length: int = 220
    max_retries: int = 2


class FishTTSClient(SpeechClient):
    def __init__(self, cfg: FishTTSConfig) -> None:
        self._cfg = cfg

    def transcribe(self, wav_path: str, language: Optional[str] = None) -> str:
        raise NotImplementedError("FishTTSClient only supports text-to-speech")

    def translate_to_uzbek(self, text: str) -> str:
        raise NotImplementedError("FishTTSClient only supports text-to-speech")

    def tts(
        self,
        text: str,
        *,
        speed: Optional[float] = None,
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
        emotion: Optional[str] = None,
        reference_audio: Optional[bytes] = None,  # ignored — uses reference_id
        reference_text: str = "",
    ) -> bytes:
        cleaned = text.strip()
        if not cleaned:
            return b""
        reference_id = (voice or self._cfg.default_reference_id).strip()
        if not reference_id:
            raise RuntimeError("Fish TTS requires a reference_id/model id")

        tagged_text = self._build_text(
            text=cleaned,
            instructions=instructions,
            emotion=emotion,
        )
        prosody_speed = self._normalize_speed(speed)
        payload = {
            "text": tagged_text,
            "reference_id": reference_id,
            "temperature": self._cfg.temperature,
            "top_p": self._cfg.top_p,
            "prosody": {
                "speed": prosody_speed,
                "volume": 0,
                "normalize_loudness": True,
            },
            "chunk_length": self._cfg.chunk_length,
            "min_chunk_length": 80,
            "normalize": True,
            "format": self._cfg.format,
            "sample_rate": self._cfg.sample_rate,
            "latency": self._cfg.latency,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.15,
            "condition_on_previous_chunks": True,
            "early_stop_threshold": 1,
        }
        return self._synthesize(payload)

    def _build_text(
        self,
        *,
        text: str,
        instructions: Optional[str],
        emotion: Optional[str],
    ) -> str:
        parts: list[str] = []
        tag = EMOTION_TAGS.get(emotion or "", "")
        if tag:
            parts.append(tag)
        if instructions:
            extra = self._instruction_to_tags(instructions)
            if extra:
                parts.append(extra)
        parts.append(text)
        return " ".join(part.strip() for part in parts if part.strip())

    def _instruction_to_tags(self, instructions: str) -> str:
        value = instructions.lower()
        tags: list[str] = []
        if "anger" in value or "forceful" in value:
            tags.append("[angry]")
        if "sad" in value:
            tags.append("[sad]")
        if "cry" in value or "sobb" in value:
            tags.append("[sobbing]")
        if "fear" in value or "trembling" in value:
            tags.append("[scared]")
        if "surprise" in value or "amazement" in value:
            tags.append("[surprised]")
        if "joy" in value or "happy" in value:
            tags.append("[happy]")
        if "laugh" in value:
            tags.append("[laughing]")
        if "cold" in value or "detachment" in value:
            tags.append("[calm]")
        if "hero" in value or "inspiring" in value:
            tags.append("[determined]")
        if "villain" in value or "menacing" in value:
            tags.append("[disdainful]")
        return " ".join(dict.fromkeys(tags))

    def _normalize_speed(self, speed: Optional[float]) -> float:
        if speed is None:
            return 1.0
        # Keep character delivery stable and let downstream timing fit do the rest.
        return max(0.94, min(1.06, float(speed)))

    def _synthesize(self, payload: dict) -> bytes:
        url = f"{self._cfg.base_url.rstrip('/')}/v1/tts"
        headers = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Content-Type": "application/json",
            "model": self._cfg.model,
        }
        last_error: Optional[Exception] = None
        for attempt in range(self._cfg.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=90,
                )
                if response.status_code == 429 and attempt < self._cfg.max_retries:
                    retry_after = self._extract_retry_after(response)
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                return response.content
            except Exception as exc:
                last_error = exc
                if attempt >= self._cfg.max_retries:
                    break
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"Fish TTS failed: {last_error}") from last_error

    def _extract_retry_after(self, response: requests.Response) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(1.0, float(retry_after))
            except ValueError:
                pass
        try:
            payload = response.json()
        except Exception:
            return 5.0
        message = str(payload)
        marker = "Please retry in "
        if marker in message:
            tail = message.split(marker, 1)[1]
            token = tail.split("s", 1)[0].strip()
            try:
                return max(1.0, float(token))
            except ValueError:
                return 5.0
        return 5.0
