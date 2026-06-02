from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class SpeechClient(ABC):
    @abstractmethod
    def transcribe(self, wav_path: str, language: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def translate_to_uzbek(self, text: str) -> str:
        pass

    def translate_batch_to_uzbek(self, texts: list[str]) -> list[str]:
        """
        Optional batch translation. Cloud clients (Naga, Google) default to
        a sequential fallback to keep their implementation simple.
        Local clients with a real NLLB model override this for speed.
        """
        return [self.translate_to_uzbek(t) for t in texts]

    @abstractmethod
    def tts(
        self,
        text: str,
        *,
        speed: Optional[float] = None,
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
        emotion: Optional[str] = None,
        # Voice-cloning support (used by local Fish Speech client).
        # Cloud clients (Naga, Google, Fish API) silently ignore these.
        reference_audio: Optional[bytes] = None,
        reference_text: str = "",
        **kwargs,
    ) -> bytes:
        pass
