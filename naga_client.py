from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

import requests
from openai import OpenAI

from client_interface import SpeechClient


@dataclass(frozen=True)
class NagaClientConfig:
    base_url: str
    api_key: str
    asr_model: str
    text_model: str
    tts_model: str
    tts_voice: str
    tts_response_format: str


class NagaClient(SpeechClient):
    def __init__(self, cfg: NagaClientConfig) -> None:
        self._cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def transcribe(self, wav_path: str, language: Optional[str] = None) -> str:
        # Try naga.ac API first
        try:
            with open(wav_path, "rb") as audio_file:
                resp = self._client.audio.transcriptions.create(
                    model=self._cfg.asr_model,
                    file=audio_file,
                    language=language,
                )
                text = getattr(resp, "text", None)
                if not text:
                    raise RuntimeError("Transcription failed: empty response text")
                return text.strip()
        except Exception as e:
            print(f"Naga API transcription failed: {str(e)}")
            print("Falling back to local Whisper")
            return self._fallback_whisper(wav_path, language)
    
    def _fallback_whisper(self, wav_path: str, language: Optional[str] = None) -> str:
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(wav_path, language=language)
            return result["text"].strip()
        except ImportError:
            raise RuntimeError("Whisper not installed. Run: pip install openai-whisper")
        except Exception as e:
            raise RuntimeError(f"Whisper fallback failed: {str(e)}")

    def translate_to_uzbek(self, text: str) -> str:
        if not text.strip():
            return ""
        resp = self._client.chat.completions.create(
            model=self._cfg.text_model,
            messages=[
                {
                    "role": "system",
                    "content": "Translate the user text into Uzbek (Latin). Keep meaning, keep names, keep punctuation. Output only the translation.",
                },
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out

    def tts(
        self,
        text: str,
        *,
        speed=None,
        voice=None,
        instructions=None,
        emotion=None,
        reference_audio: Optional[bytes] = None,  # ignored by cloud client
        reference_text: str = "",
        **extra_kwargs,
    ) -> bytes:
        if not text.strip():
            return b""
        # Build base parameters
        params = {
            "model": self._cfg.tts_model,
            "voice": voice or self._cfg.tts_voice,
            "input": text,
            "response_format": self._cfg.tts_response_format,
        }
        if speed is not None:
            params["speed"] = speed
        
        # Try different approaches
        try:
            # First try with both speed and instructions (if provided)
            test_params = params.copy()
            if instructions:
                test_params["instructions"] = instructions
            test_params.update(extra_kwargs)
            return self._client.audio.speech.create(**test_params).read()
        except TypeError:
            # If instructions failed, try without
            test_params = params.copy()
            test_params.update(extra_kwargs)
            return self._client.audio.speech.create(**test_params).read()
