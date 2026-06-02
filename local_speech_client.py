"""
local_speech_client.py
──────────────────────
Full local SpeechClient implementation:
  - ASR      : faster-whisper (local, fast)
  - Translate : NLLB-200 (local, Meta's model, supports Uzbek)
  - TTS      : Fish Speech 1.5 (local, S2 open-source, voice cloning)

Voice cloning:
  Pass reference_audio (WAV bytes) + reference_text to tts().
  Fish Speech will clone the voice of the reference speaker — no presets needed.
  Each character/speaker in dubbing.py gets their original audio as reference,
  so every character sounds exactly like their original voice, only in Uzbek.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import List, Optional

from client_interface import SpeechClient
from local_fish_tts import (
    EMOTION_TAGS,
    FishSpeechServer,
    ensure_fish_speech_installed,
    ensure_model_downloaded,
)
from local_translator import LocalTranslator

logger = logging.getLogger(__name__)


def _pip(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


# ── Whisper ASR helper ────────────────────────────────────────────────────────

class _LocalWhisper:
    """Lazy-loaded faster-whisper ASR (falls back to openai-whisper)."""

    _instance: Optional["_LocalWhisper"] = None

    def __init__(self, model_size: str = "large-v3") -> None:
        self._model_size = model_size
        self._model = None

    @classmethod
    def get(cls, model_size: str = "large-v3") -> "_LocalWhisper":
        if cls._instance is None:
            cls._instance = cls(model_size)
        return cls._instance

    def transcribe(self, wav_path: str, language: Optional[str] = None) -> str:
        model = self._load()
        if model is None:
            return ""
        return self._run(model, wav_path, language)

    def transcribe_with_word_timestamps(
        self, wav_path: str, language: Optional[str] = None
    ) -> List[dict]:
        """
        Return a list of word-level dicts: {word, start_ms, end_ms, prob}.
        Used by the dubbing pipeline to reconstruct per-segment timing
        after translation so TTS output matches the original cadence.
        """
        model = self._load()
        if model is None:
            return []
        kind, m = model
        if kind == "faster":
            segments, _ = m.transcribe(
                wav_path,
                language=language,
                beam_size=5,
                word_timestamps=True,
            )
            words: List[dict] = []
            for seg in segments:
                if not getattr(seg, "words", None):
                    continue
                for w in seg.words:
                    if w.start is None or w.end is None:
                        continue
                    words.append(
                        {
                            "word": w.word,
                            "start_ms": int(w.start * 1000),
                            "end_ms": int(w.end * 1000),
                            "prob": float(getattr(w, "probability", 1.0)),
                        }
                    )
            return words
        # openai-whisper fallback: no per-word timestamps
        result = m.transcribe(wav_path, language=language) if language else m.transcribe(wav_path)
        return [
            {
                "word": w["word"],
                "start_ms": int(w["start"] * 1000),
                "end_ms": int(w["end"] * 1000),
                "prob": float(w.get("probability", 1.0)),
            }
            for w in result.get("words", [])
        ]

    def _load(self):
        if self._model is not None:
            return self._model
        # Try faster-whisper first (much faster than openai-whisper)
        try:
            from faster_whisper import WhisperModel  # type: ignore
            self._model = ("faster", WhisperModel(self._model_size, compute_type="auto"))
            logger.info("Loaded faster-whisper %s", self._model_size)
            return self._model
        except ImportError:
            logger.info("faster-whisper not installed, installing…")
            try:
                _pip("faster-whisper")
                from faster_whisper import WhisperModel  # type: ignore
                self._model = ("faster", WhisperModel(self._model_size, compute_type="auto"))
                return self._model
            except Exception as e:
                logger.warning("faster-whisper failed (%s), falling back to openai-whisper", e)

        # openai-whisper fallback
        try:
            import whisper  # type: ignore
        except ImportError:
            _pip("openai-whisper")
            import whisper  # type: ignore
        m = whisper.load_model("large" if "large" in self._model_size else self._model_size)
        self._model = ("openai", m)
        logger.info("Loaded openai-whisper")
        return self._model

    def _run(self, model_tuple, wav_path: str, language: Optional[str]) -> str:
        kind, model = model_tuple
        if kind == "faster":
            segments, _ = model.transcribe(wav_path, language=language, beam_size=5)
            return " ".join(seg.text for seg in segments).strip()
        else:
            kwargs = {"language": language} if language else {}
            result = model.transcribe(wav_path, **kwargs)
            return result["text"].strip()


# ── Main client ───────────────────────────────────────────────────────────────

class LocalSpeechClient(SpeechClient):
    """
    Fully local speech pipeline.

    model_dir : directory where Fish Speech checkpoints are stored.
    whisper_size : faster-whisper model size (tiny/base/small/medium/large-v3).
    device : "auto" (default) | "cuda" | "cpu"
    """

    def __init__(
        self,
        model_dir: str,
        whisper_size: str = "large-v3",
        device: str = "auto",
        translator_cache_dir: Optional[str] = None,
    ) -> None:
        self._model_dir = model_dir
        self._whisper_size = whisper_size
        self._device = device
        self._translator_cache_dir = translator_cache_dir
        self._server: Optional[FishSpeechServer] = None
        self._translator: Optional[LocalTranslator] = None

        # Setup Fish Speech (install + download model) eagerly so the user
        # sees a clear log trail instead of a silent delay on first request.
        ensure_fish_speech_installed()
        ensure_model_downloaded(model_dir)

    # ── SpeechClient interface ────────────────────────────────────────────────

    def transcribe(self, wav_path: str, language: Optional[str] = None) -> str:
        w = _LocalWhisper.get(self._whisper_size)
        text = w.transcribe(wav_path, language=language)
        logger.debug("Transcribed: %s chars", len(text))
        return text

    def transcribe_with_word_timestamps(
        self, wav_path: str, language: Optional[str] = None
    ) -> List[dict]:
        """
        Word-level ASR with millisecond timestamps. Used to estimate the
        original speaking rate so the dubbed TTS can match the same tempo
        (per-word pauses, syllable rate, etc).
        """
        w = _LocalWhisper.get(self._whisper_size)
        return w.transcribe_with_word_timestamps(wav_path, language=language)

    def translate_to_uzbek(self, text: str) -> str:
        t = self._get_translator()
        result = t.translate(text)
        logger.debug("Translated → uz: %s chars", len(result))
        return result

    def translate_batch_to_uzbek(self, texts: list[str]) -> list[str]:
        t = self._get_translator()
        results = t.translate_batch(texts)
        logger.debug("Batch translated → uz: %s items", len(results))
        return results

    def tts(
        self,
        text: str,
        *,
        speed: Optional[float] = None,
        voice: Optional[str] = None,           # ignored for local — uses reference_audio
        instructions: Optional[str] = None,    # optional free-text style hint
        emotion: Optional[str] = None,
        reference_audio: Optional[bytes] = None,
        reference_text: str = "",
        extra_references: Optional[list[dict]] = None,
        **kwargs,
    ) -> bytes:
        text = text.strip()
        if not text:
            return b""

        # Build the prompt: emotion tag + optional style hint + text
        parts: list[str] = []
        if emotion and emotion in EMOTION_TAGS:
            parts.append(EMOTION_TAGS[emotion])
        if instructions:
            tag = _instructions_to_fish_tag(instructions)
            if tag:
                parts.append(tag)
        parts.append(text)
        final_text = " ".join(p.strip() for p in parts if p.strip())

        server = self._get_server()

        refs: list[dict] = []
        if reference_audio:
            refs.append({"audio": reference_audio, "text": reference_text or ""})
        if extra_references:
            refs.extend(extra_references)

        # Temperature/top_p: slightly more expressive for emotional content
        temperature = 0.75 if emotion else 0.7
        top_p = 0.8 if emotion else 0.7

        return server.synthesize(
            text=final_text,
            references=refs,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=1.2,
            max_new_tokens=1024,
            chunk_length=200,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_server(self) -> FishSpeechServer:
        if self._server is None:
            self._server = FishSpeechServer.get_instance(
                model_dir=self._model_dir,
                device=self._device,
            )
        return self._server

    def _get_translator(self) -> LocalTranslator:
        if self._translator is None:
            self._translator = LocalTranslator.get_instance(self._translator_cache_dir)
        return self._translator


# ── Instruction → Fish tag mapping ───────────────────────────────────────────

def _instructions_to_fish_tag(instructions: str) -> str:
    v = instructions.lower()
    tags: list[str] = []
    if any(w in v for w in ("anger", "angry", "forceful", "shout")):
        tags.append("[angry]")
    if "sad" in v:
        tags.append("[sad]")
    if any(w in v for w in ("cry", "sob", "weep")):
        tags.append("[sobbing]")
    if any(w in v for w in ("fear", "tremble", "scared")):
        tags.append("[scared]")
    if any(w in v for w in ("surprise", "amazement")):
        tags.append("[surprised]")
    if any(w in v for w in ("joy", "happy", "cheerful")):
        tags.append("[happy]")
    if "laugh" in v:
        tags.append("[laughing]")
    if any(w in v for w in ("calm", "cold", "detach")):
        tags.append("[calm]")
    if any(w in v for w in ("hero", "inspire", "confident")):
        tags.append("[determined]")
    if any(w in v for w in ("villain", "menac", "dark")):
        tags.append("[disdainful]")
    return " ".join(dict.fromkeys(tags))
