"""
local_fish_tts.py
─────────────────
Local Fish Speech 1.5 (S2 open-source) server manager with zero-shot voice cloning.

Features:
- Auto-installs fish-speech package if missing
- Auto-downloads model weights from HuggingFace (fishaudio/fish-speech-1.5)
- Starts a local HTTP API server as a background subprocess (singleton)
- Voice cloning: pass any reference WAV → clone that voice
- Natural, emotion-aware speech via prompt engineering
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────────────────────
FISH_REPO_ID = os.getenv("FISH_LOCAL_HF_REPO", "fishaudio/fish-speech-1.5")
FISH_DECODER_CONFIG_NAME = os.getenv("FISH_LOCAL_DECODER_CONFIG_NAME", "firefly_gan_vq")
FISH_SAMPLE_RATE = 44100

# Emotion tags supported by Fish Speech S2 model
EMOTION_TAGS: dict[str, str] = {
    "gazab":          "[angry] [shouting]",
    "qaygu":          "[sad]",
    "yiglash":        "[sobbing] [sad]",
    "qorquv":         "[scared] [panting]",
    "hayrat":         "[surprised]",
    "xursandchilik":  "[happy]",
    "kulish":         "[laughing]",
    "sovuqqonlik":    "[calm]",
    "qahramonona":    "[confident] [determined]",
    "yovuz_qahramon": "[disdainful] [confident]",
}

# ── Package helpers ───────────────────────────────────────────────────────────

def _pip_install(*packages: str, quiet: bool = True) -> None:
    cmd = [sys.executable, "-m", "pip", "install"] + list(packages)
    if quiet:
        cmd += ["-q"]
    subprocess.check_call(cmd)


def _get_requests():
    """Lazy import requests — install if missing."""
    try:
        import requests as _req
        return _req
    except ImportError:
        _pip_install("requests")
        import requests as _req
        return _req


def ensure_fish_speech_installed() -> None:
    """Install fish-speech package if not already present."""
    try:
        import fish_speech  # noqa: F401
        return
    except ImportError:
        pass

    logger.info("fish-speech not found — installing (this may take a few minutes)...")
    # Try PyPI first
    try:
        _pip_install("fish-speech", quiet=False)
        import fish_speech  # noqa: F401
        logger.info("fish-speech installed via pip")
        return
    except Exception as e:
        logger.warning("pip install fish-speech failed (%s), trying from GitHub...", e)

    # Try from GitHub main branch
    try:
        _pip_install(
            "git+https://github.com/fishaudio/fish-speech.git@main",
            quiet=False,
        )
        logger.info("fish-speech installed from GitHub")
    except Exception as e2:
        raise RuntimeError(
            "Could not install fish-speech automatically.\n"
            "Please run manually:  pip install fish-speech\n"
            f"Error: {e2}"
        ) from e2


def ensure_model_downloaded(model_dir: str) -> None:
    """Download Fish Speech model weights from HuggingFace if not present."""
    path = Path(model_dir)
    sentinel_files = [
        "codec.pth",
        "model.pth",
        "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        "config.json",
    ]
    if any((path / f).exists() for f in sentinel_files):
        logger.info("Fish Speech model found at %s", model_dir)
        return

    logger.info("Downloading Fish Speech model (%s) to %s ...", FISH_REPO_ID, model_dir)
    path.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _pip_install("huggingface_hub")
        from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=FISH_REPO_ID,
        local_dir=model_dir,
        ignore_patterns=["*.git*"],
    )
    logger.info("Fish Speech model downloaded to %s", model_dir)


def _find_decoder_checkpoint(model_dir: str) -> Optional[str]:
    path = Path(model_dir)
    candidates = [
        path / "codec.pth",
        path / "decoder.pth",
        path / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    matches = list(path.rglob("*codec*.pth")) + list(path.rglob("*vq*.pth"))
    return str(matches[0]) if matches else None


# ── Port helper ───────────────────────────────────────────────────────────────

def _find_free_port(start: int = 18080) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free TCP port found in range 18080-18280")


# ── Server manager ────────────────────────────────────────────────────────────

class FishSpeechServer:
    """
    Singleton wrapper around the local Fish Speech API server subprocess.
    Started once, kept alive for the lifetime of the process.
    """

    _instance: Optional["FishSpeechServer"] = None
    _lock = threading.Lock()

    def __init__(self, model_dir: str, port: int = 0, device: str = "auto") -> None:
        self.model_dir = model_dir
        self.port = port or _find_free_port()
        self.device = self._resolve_device(device)
        self._proc: Optional[subprocess.Popen] = None
        atexit.register(self.stop)

    # ── Singleton ─────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        model_dir: str,
        port: int = 0,
        device: str = "auto",
    ) -> "FishSpeechServer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = cls(model_dir, port, device)
                    inst.start()
                    cls._instance = inst
        return cls._instance

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return  # already running

        decoder_checkpoint = _find_decoder_checkpoint(self.model_dir)
        logger.info(
            "Starting Fish Speech server (port=%s, device=%s, model=%s, decoder=%s)...",
            self.port, self.device, self.model_dir, decoder_checkpoint,
        )

        # Command candidates tried in order
        cmds = [
            [
                sys.executable, "-m", "tools.api_server",
                "--listen", f"127.0.0.1:{self.port}",
                "--llama-checkpoint-path", self.model_dir,
                "--decoder-checkpoint-path", decoder_checkpoint or os.path.join(self.model_dir, "codec.pth"),
                "--decoder-config-name", FISH_DECODER_CONFIG_NAME,
            ],
            [
                sys.executable, "tools/api_server.py",
                "--listen", f"127.0.0.1:{self.port}",
                "--llama-checkpoint-path", self.model_dir,
                "--decoder-checkpoint-path", decoder_checkpoint or os.path.join(self.model_dir, "codec.pth"),
                "--decoder-config-name", FISH_DECODER_CONFIG_NAME,
            ],
            [
                sys.executable, "-m", "fish_speech.cli.server",
                "--checkpoint", self.model_dir,
                "--device", self.device,
                "--listen", f"127.0.0.1:{self.port}",
            ],
            [
                sys.executable, "-m", "fish_speech.cli.app",
                "--checkpoint-path", self.model_dir,
                "--device", self.device,
                "--server-port", str(self.port),
                "--no-browser",
            ],
            [
                sys.executable, "-m", "tools.api_server",
                "--listen", f"0.0.0.0:{self.port}",
            ],
        ]

        for cmd in cmds:
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if self._wait_ready(timeout=180):
                    logger.info("Fish Speech server ready on port %s", self.port)
                    return
                # Server didn't respond — kill and try next cmd
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
                except Exception:
                    pass
                self._proc = None
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning("Server start attempt failed (%s), trying next...", e)
                self._proc = None

        raise RuntimeError(
            "Could not start Fish Speech API server.\n"
            "Make sure fish-speech is installed: pip install fish-speech"
        )

    def stop(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
            logger.info("Fish Speech server stopped")

    def _legacy_synthesize(
        self,
        text: str,
        references: Optional[list] = None,
        fmt: str = "wav",
        top_p: float = 0.7,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 1024,
        chunk_length: int = 200,
    ) -> bytes:
        """POST to /v1/tts and return raw WAV bytes."""
        req = _get_requests()

        payload: dict = {
            "text": text,
            "format": fmt,
            "references": references or [],
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "chunk_length": chunk_length,
            "normalize": True,
            "latency": "normal",
        }

        # Try msgpack (binary-safe) first → JSON fallback
        try:
            import ormsgpack  # type: ignore
            body = ormsgpack.packb(payload, option=ormsgpack.OPT_SERIALIZE_NUMPY)
            headers = {"Content-Type": "application/msgpack"}
        except ImportError:
            # Encode audio bytes as base64 for JSON transport
            import base64
            safe_payload = json.loads(json.dumps(payload, default=str))
            for ref in safe_payload.get("references", []):
                if isinstance(ref.get("audio"), (bytes, bytearray)):
                    ref["audio"] = base64.b64encode(ref["audio"]).decode()
            body = json.dumps(safe_payload).encode()
            headers = {"Content-Type": "application/json"}

        url = f"http://127.0.0.1:{self.port}/v1/tts"
        resp = req.post(url, content=body, headers=headers, timeout=180)
        resp.raise_for_status()
        return resp.content

    # ── Internals ─────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        references: Optional[list] = None,
        fmt: str = "wav",
        top_p: float = 0.7,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 1024,
        chunk_length: int = 200,
    ) -> bytes:
        req = _get_requests()
        payload: dict = {
            "text": text,
            "format": fmt,
            "references": references or [],
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "chunk_length": chunk_length,
            "normalize": True,
            "latency": "normal",
        }
        url = f"http://127.0.0.1:{self.port}/v1/tts"
        last_error: Optional[Exception] = None

        try:
            resp = req.post(url, json=self._json_payload(payload, references or []), timeout=180)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            last_error = exc

        try:
            import ormsgpack  # type: ignore
            body = ormsgpack.packb(payload, option=ormsgpack.OPT_SERIALIZE_NUMPY)
            resp = req.post(
                url,
                content=body,
                headers={"Content-Type": "application/msgpack"},
                timeout=180,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            last_error = exc

        try:
            safe_payload = self._json_payload(payload, references or [])
            resp = req.post(
                url,
                content=json.dumps(safe_payload).encode(),
                headers={"Content-Type": "application/json"},
                timeout=180,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            raise RuntimeError(f"Fish Speech TTS request failed: {exc}") from last_error

    def _json_payload(self, payload: dict, references: list) -> dict:
        safe_payload = {key: value for key, value in payload.items() if key != "references"}
        if references:
            first_ref = references[0]
            audio = first_ref.get("audio")
            if isinstance(audio, (bytes, bytearray)):
                safe_payload["reference_audio"] = base64.b64encode(audio).decode()
            safe_payload["reference_text"] = str(first_ref.get("text") or "")
        return safe_payload

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _wait_ready(self, timeout: int = 180) -> bool:
        """Poll /v1/health until the server responds or timeout expires."""
        req = _get_requests()
        deadline = time.time() + timeout
        health_urls = [
            f"http://127.0.0.1:{self.port}/v1/health",
            f"http://127.0.0.1:{self.port}/health",
        ]
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                try:
                    stderr_out = self._proc.stderr.read().decode(errors="replace")
                except Exception:
                    stderr_out = ""
                logger.error(
                    "Fish Speech server process exited early. stderr:\n%s",
                    stderr_out[-1000:],
                )
                return False
            for url in health_urls:
                try:
                    r = req.get(url, timeout=2)
                    if r.status_code < 500:
                        return True
                except Exception:
                    pass
            time.sleep(3)
        return False
