from __future__ import annotations

from typing import Optional

from client_interface import SpeechClient
from naga_client import NagaClient, NagaClientConfig
from fish_tts_client import FishTTSClient, FishTTSConfig
from google_tts_client import GoogleTTSClient, GoogleTTSConfig


class HybridClient(SpeechClient):
    """
    Hybrid client that uses:
    - NAGA for ASR (speech-to-text) and Translation
    - Google/Gemini for TTS (text-to-speech) with emotions
    - NAGA TTS as fallback if Google fails!
    """
    
    def __init__(
        self,
        naga_config: NagaClientConfig,
        google_config: Optional[GoogleTTSConfig] = None,
        fish_config: Optional[FishTTSConfig] = None,
    ):
        self.naga_client = NagaClient(naga_config)
        if fish_config is not None:
            self.tts_client: SpeechClient = FishTTSClient(fish_config)
        elif google_config is not None:
            self.tts_client = GoogleTTSClient(google_config)
        else:
            raise RuntimeError("HybridClient requires a TTS client config")
        
    def transcribe(self, wav_path: str, language: Optional[str] = None) -> str:
        return self.naga_client.transcribe(wav_path, language)
        
    def translate_to_uzbek(self, text: str) -> str:
        return self.naga_client.translate_to_uzbek(text)
        
    def tts(
        self,
        text: str,
        *,
        speed: Optional[float] = None,
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
        emotion: Optional[str] = None,
        reference_audio: Optional[bytes] = None,
        reference_text: str = "",
    ) -> bytes:
        # Try configured TTS provider first
        try:
            return self.tts_client.tts(
                text,
                speed=speed,
                voice=voice,
                instructions=instructions,
                emotion=emotion,
                reference_audio=reference_audio,
                reference_text=reference_text,
            )
        except Exception:
            # Fallback to NAGA TTS if primary provider fails!
            return self.naga_client.tts(text, speed=speed, instructions=instructions)
