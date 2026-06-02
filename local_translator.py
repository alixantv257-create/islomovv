"""
local_translator.py
───────────────────
Fully local Uzbek translation using Meta's NLLB-200 model.

Model: facebook/nllb-200-distilled-600M  (~1.5 GB, supports 200 languages)
Target language: uzb_Latn  (Uzbek Latin script)

Auto-installs transformers + sentencepiece on first use.
Model is cached to disk (HuggingFace cache or LOCAL_TRANSLATOR_MODEL_DIR).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from typing import Optional
import config

logger = logging.getLogger(__name__)

# HuggingFace model ID
_NLLB_MODEL = config.LOCAL_NLLB_MODEL
# NLLB language code for Uzbek (Latin)
_UZB_LANG = "uzb_Latn"

# Reasonable max token lengths
_MAX_SRC_LEN = 512
_MAX_TGT_LEN = 600


def _pip(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


class LocalTranslator:
    """
    Singleton wrapper around NLLB-200-distilled-600M for Uzbek translation.

    Lazy-loads the model on first call so startup is fast.
    Thread-safe.
    """

    _instance: Optional["LocalTranslator"] = None
    _init_lock = threading.Lock()

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self._model_dir = model_dir
        self._tokenizer = None
        self._model = None
        self._device: Optional[str] = None
        self._ready = False
        self._load_lock = threading.Lock()

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls, model_dir: Optional[str] = None) -> "LocalTranslator":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls(model_dir)
        return cls._instance

    # ── Public API ────────────────────────────────────────────────────────────

    def translate(self, text: str, src_lang: Optional[str] = None) -> str:
        """
        Translate *text* to Uzbek (Latin script).

        src_lang: NLLB language code (e.g. "rus_Cyrl", "eng_Latn").
                  Pass None to auto-detect (slightly slower).
        """
        text = text.strip()
        if not text:
            return ""

        self._ensure_loaded()

        import torch

        # Auto-detect source language if not provided
        if src_lang is None:
            src_lang = self._detect_lang(text)

        tokenizer = self._tokenizer
        model = self._model
        device = self._device

        # Force source language token
        tokenizer.src_lang = src_lang
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_MAX_SRC_LEN,
        ).to(device)

        forced_bos = tokenizer.convert_tokens_to_ids(_UZB_LANG)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=_MAX_TGT_LEN,
                num_beams=4,
                early_stopping=True,
            )

        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return result.strip()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._ready:
            return
        with self._load_lock:
            if self._ready:
                return
            self._load_model()
            self._ready = True

    def _load_model(self) -> None:
        # Make sure packages are installed
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401
        except ImportError:
            logger.info("Installing transformers + sentencepiece …")
            _pip("transformers", "sentencepiece", "sacremoses")
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        kwargs: dict = {}
        if self._model_dir:
            kwargs["cache_dir"] = self._model_dir

        logger.info("Loading NLLB-200 translation model … (first run downloads ~1.5 GB)")

        self._tokenizer = AutoTokenizer.from_pretrained(_NLLB_MODEL, **kwargs)

        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                _NLLB_MODEL,
                **kwargs,
            ).to(self._device)
        except Exception:
            # Fallback to CPU without torch device management
            self._device = "cpu"
            self._model = AutoModelForSeq2SeqLM.from_pretrained(_NLLB_MODEL, **kwargs)

        logger.info("NLLB-200 model loaded (device=%s)", self._device)

    def _detect_lang(self, text: str) -> str:
        """
        Fast heuristic language detection without an external library.
        Checks script (Cyrillic → Russian, Latin → English as default).
        """
        cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
        arabic   = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
        korean   = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
        japanese = sum(1 for c in text if "\u3040" <= c <= "\u30ff")
        chinese  = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        total    = max(1, len(text))

        if cyrillic / total > 0.3:
            return "rus_Cyrl"
        if arabic / total > 0.3:
            return "arb_Arab"
        if korean / total > 0.3:
            return "kor_Hang"
        if (japanese + chinese) / total > 0.3:
            return "jpn_Jpan"
        # Default assumption: English (Latin)
        return "eng_Latn"

    def translate_batch(self, texts: list[str], src_lang: Optional[str] = None) -> list[str]:
        """
        Translate a list of strings in a single batched model.generate() call.
        Much faster than calling translate() N times because GPU kernel
        launches and tokenizer overhead are amortized.

        Empty strings in the input are returned as empty strings (no model call).
        A single src_lang is used for the entire batch (auto-detected if None).
        """
        texts = [t.strip() for t in texts]
        if not texts:
            return []

        # Preserve an index so we can return "" for empty inputs without
        # spending model cycles on them.
        indexed = [(i, t) for i, t in enumerate(texts) if t]
        if not indexed:
            return [""] * len(texts)

        self._ensure_loaded()

        import torch

        if src_lang is None:
            # Detect from the first non-empty text
            src_lang = self._detect_lang(indexed[0][1])

        indices = [i for i, _ in indexed]
        payload = [t for _, t in indexed]

        tokenizer = self._tokenizer
        model = self._model
        device = self._device

        tokenizer.src_lang = src_lang
        enc = tokenizer(
            payload,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_MAX_SRC_LEN,
        ).to(device)

        forced_bos = tokenizer.convert_tokens_to_ids(_UZB_LANG)

        with torch.no_grad():
            output_ids = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                max_length=_MAX_TGT_LEN,
                num_beams=4,
                early_stopping=True,
            )

        decoded = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in output_ids]

        results: list[str] = [""] * len(texts)
        for idx, out in zip(indices, decoded):
            results[idx] = out
        return results
