from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Sequence

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")
    return ffmpeg


def run(args: Sequence[str]) -> None:
    logger.info("Running command: %s", " ".join(args))
    proc = subprocess.run(
        list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        logger.error("Command failed with exit_code=%s", proc.returncode)
        raise RuntimeError(proc.stdout[-4000:])
    if proc.stdout.strip():
        logger.info("Command output:\n%s", proc.stdout[-2000:])
