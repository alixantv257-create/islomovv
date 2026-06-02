from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import webrtcvad
from pydub import AudioSegment

from ffmpeg_utils import ensure_dir, require_ffmpeg, run
from client_interface import SpeechClient

logger = logging.getLogger(__name__)
_SPEAKER_ENCODER = None
_SPEAKER_ENCODER_FAILED = False
_DENOISE_FAILED = False
_LOUDNORM_FAILED = False


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    wav_path: str
    pcm16: Optional[bytes] = None
    sample_rate: int = 16000
    extras: Dict[str, object] = field(default_factory=dict)


def _write_wav(path: str, pcm16: bytes, *, sample_rate: int) -> None:
    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)


def _read_wav_mono16(path: str) -> Tuple[bytes, int]:
    with contextlib.closing(wave.open(path, "rb")) as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise RuntimeError("VAD input must be mono 16-bit PCM WAV")
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, sample_rate


def _read_wav_float(path: str) -> Tuple[np.ndarray, int]:
    pcm, sample_rate = _read_wav_mono16(path)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sample_rate


def _pcm16_to_audiosegment(pcm: bytes, sample_rate: int) -> AudioSegment:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return AudioSegment(
        (audio * 32767.0).astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )


def _slice_wav_segment(
    source_path: str,
    out_path: str,
    *,
    start_ms: int,
    end_ms: int,
) -> None:
    audio = AudioSegment.from_file(source_path)
    clipped = audio[max(0, start_ms) : max(start_ms, end_ms)]
    clipped.export(out_path, format="wav")


def _frame_generator(pcm: bytes, sample_rate: int, frame_ms: int) -> Iterable[Tuple[bytes, int]]:
    bytes_per_sample = 2
    frame_size = int(sample_rate * (frame_ms / 1000.0) * bytes_per_sample)
    offset = 0
    t_ms = 0
    while offset + frame_size <= len(pcm):
        yield pcm[offset : offset + frame_size], t_ms
        offset += frame_size
        t_ms += frame_ms


def _materialize_segments(segments: List[Segment], out_dir: str) -> List[Segment]:
    """
    Persist any in-memory segments to disk and return the same list with wav_path
    set. Idempotent: if wav_path already exists on disk it's left as-is.
    """
    ensure_dir(out_dir)
    for idx, seg in enumerate(segments):
        if seg.wav_path and os.path.exists(seg.wav_path):
            continue
        if seg.pcm16 is None:
            continue
        out_path = os.path.join(out_dir, f"seg_{idx:04d}.wav")
        _write_wav(out_path, seg.pcm16, sample_rate=seg.sample_rate)
        seg.wav_path = out_path
    return segments


def _vad_segments_in_memory(
    pcm: bytes,
    sample_rate: int,
    *,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    padding_ms: int = 300,
    join_gap_ms: int = 250,
) -> List[Segment]:
    """
    VAD that returns segments in RAM as (start_ms, end_ms, pcm16) tuples.
    No disk I/O is performed here — the caller decides when to write.
    """
    if sample_rate not in (8000, 16000, 32000, 48000):
        raise RuntimeError(f"Unsupported sample rate for VAD: {sample_rate}")

    vad = webrtcvad.Vad(aggressiveness)
    frames = list(_frame_generator(pcm, sample_rate, frame_ms))
    triggered = False
    voiced: List[Tuple[bytes, int]] = []
    raw: List[Segment] = []
    padding_frames = max(1, padding_ms // frame_ms)
    ring: List[Tuple[bytes, int, bool]] = []

    def flush() -> None:
        nonlocal voiced, triggered
        if not voiced:
            triggered = False
            ring.clear()
            return
        start_ms = voiced[0][1]
        end_ms = voiced[-1][1] + frame_ms
        pcm_out = b"".join(b for b, _t in voiced)
        raw.append(Segment(start_ms=start_ms, end_ms=end_ms, wav_path="", pcm16=pcm_out, sample_rate=sample_rate))
        voiced = []
        triggered = False
        ring.clear()

    for frame_bytes, t_ms in frames:
        is_speech = vad.is_speech(frame_bytes, sample_rate)
        ring.append((frame_bytes, t_ms, is_speech))
        if len(ring) > padding_frames:
            ring.pop(0)

        if not triggered:
            num_voiced = sum(1 for _b, _t, s in ring if s)
            if num_voiced > (len(ring) * 0.8):
                triggered = True
                for b, t, _s in ring:
                    voiced.append((b, t))
                ring.clear()
        else:
            voiced.append((frame_bytes, t_ms))
            num_unvoiced = sum(1 for _b, _t, s in ring if not s)
            if num_unvoiced > (len(ring) * 0.8):
                flush()
    flush()

    # Merge segments whose gap is below join_gap_ms by concatenating PCM.
    merged: List[Segment] = []
    for seg in raw:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if seg.start_ms - prev.end_ms <= join_gap_ms:
            silence_ms = max(0, seg.start_ms - prev.end_ms)
            silence_pcm = b"\x00\x00" * int(sample_rate * (silence_ms / 1000.0))
            new_pcm = (prev.pcm16 or b"") + silence_pcm + (seg.pcm16 or b"")
            merged[-1] = Segment(
                start_ms=prev.start_ms,
                end_ms=seg.end_ms,
                wav_path="",
                pcm16=new_pcm,
                sample_rate=sample_rate,
            )
        else:
            merged.append(seg)
    return merged


def _vad_segments(
    wav_path: str,
    *,
    aggressiveness: int = 2,
    frame_ms: int = 30,
) -> List[Segment]:
    """
    Disk-based VAD wrapper. Reads the mono 16-bit WAV at wav_path, runs
    in-memory VAD, materializes segments under <wav_dir>/vad/ and returns
    the populated list.
    """
    pcm, sample_rate = _read_wav_mono16(wav_path)
    segments = _vad_segments_in_memory(
        pcm, sample_rate, aggressiveness=aggressiveness, frame_ms=frame_ms
    )
    out_dir = os.path.join(os.path.dirname(wav_path), "vad")
    return _materialize_segments(segments, out_dir)


def _find_silence_ranges(
    pcm: bytes,
    sample_rate: int,
    *,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    min_silence_ms: int = 300,
) -> List[Tuple[int, int]]:
    """
    Return [(start_ms, end_ms), ...] for each non-silent run in *pcm*.
    Used to find natural break points inside long VAD segments.
    """
    if sample_rate not in (8000, 16000, 32000, 48000):
        raise RuntimeError(f"Unsupported sample rate for VAD: {sample_rate}")
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(_frame_generator(pcm, sample_rate, frame_ms))
    in_speech = False
    run_start = 0
    min_silence_frames = max(1, min_silence_ms // frame_ms)
    sil_run = 0
    for frame_bytes, t_ms in frames:
        is_speech = vad.is_speech(frame_bytes, sample_rate)
        if is_speech:
            if not in_speech:
                in_speech = True
                run_start = t_ms
            sil_run = 0
        else:
            if in_speech:
                sil_run += 1
                if sil_run >= min_silence_frames:
                    yield (run_start, t_ms)
                    in_speech = False
                    sil_run = 0
    if in_speech:
        yield (run_start, frames[-1][1] + frame_ms)


def _split_long_segments_at_silence(
    segments: List[Segment],
    *,
    max_segment_ms: int = 9000,
    min_segment_ms: int = 2500,
    min_silence_ms: int = 300,
) -> List[Segment]:
    """
    Re-split segments longer than max_segment_ms at internal silence points.
    Falls back to even splitting when no silence is found inside the segment.
    """
    out: List[Segment] = []
    for seg in segments:
        seg_len = seg.end_ms - seg.start_ms
        if seg_len <= max_segment_ms:
            out.append(seg)
            continue
        if seg.pcm16 is None:
            # No PCM available — fall back to even splitting
            out.extend(_even_split(seg, max_segment_ms, min_segment_ms))
            continue

        # Restrict silence search to [seg.start_ms, seg.end_ms] of original audio.
        # For in-memory segments the PCM is exactly the segment audio, so
        # silence times are segment-local (start at 0).
        local_silences = list(
            _find_silence_ranges(
                seg.pcm16,
                seg.sample_rate,
                aggressiveness=2,
                min_silence_ms=min_silence_ms,
            )
        )
        if not local_silences:
            out.extend(_even_split(seg, max_segment_ms, min_segment_ms))
            continue

        # Silence midpoints are natural break points.
        boundaries: List[int] = [0]
        for s_ms, e_ms in local_silences:
            boundaries.append((s_ms + e_ms) // 2)
        boundaries.append(seg_len)
        # Filter to chunks ≥ min_segment_ms and ≤ max_segment_ms
        chunk_starts = [0]
        cursor = 0
        for b in boundaries[1:]:
            if b - cursor >= min_segment_ms:
                chunk_starts.append(b)
                cursor = b
        if cursor < seg_len:
            chunk_starts.append(seg_len)

        sample_rate = seg.sample_rate
        bytes_per_ms = sample_rate * 2 / 1000.0
        for i in range(len(chunk_starts) - 1):
            local_start = chunk_starts[i]
            local_end = chunk_starts[i + 1]
            if local_end - local_start > max_segment_ms:
                # Oversized chunk → even-split that part
                fake_seg = Segment(
                    start_ms=seg.start_ms + local_start,
                    end_ms=seg.start_ms + local_end,
                    wav_path="",
                    pcm16=_slice_pcm(seg.pcm16, sample_rate, local_start, local_end),
                    sample_rate=sample_rate,
                )
                out.extend(_even_split(fake_seg, max_segment_ms, min_segment_ms))
                continue
            start_byte = int(local_start * bytes_per_ms)
            end_byte = int(local_end * bytes_per_ms)
            chunk_pcm = seg.pcm16[start_byte:end_byte]
            out.append(
                Segment(
                    start_ms=seg.start_ms + local_start,
                    end_ms=seg.start_ms + local_end,
                    wav_path="",
                    pcm16=chunk_pcm,
                    sample_rate=sample_rate,
                )
            )
    return out


def _slice_pcm(pcm: bytes, sample_rate: int, start_ms: int, end_ms: int) -> bytes:
    bytes_per_ms = sample_rate * 2 / 1000.0
    start_byte = int(start_ms * bytes_per_ms)
    end_byte = int(end_ms * bytes_per_ms)
    return pcm[start_byte:end_byte]


def _even_split(seg: Segment, max_segment_ms: int, min_segment_ms: int) -> List[Segment]:
    seg_len = seg.end_ms - seg.start_ms
    if seg_len <= max_segment_ms:
        return [seg]
    chunk_count = max(2, int(np.ceil(seg_len / max_segment_ms)))
    chunk_ms = max(min_segment_ms, seg_len // chunk_count)
    out: List[Segment] = []
    cursor = seg.start_ms
    sample_rate = seg.sample_rate if seg.pcm16 else 16000
    while cursor < seg.end_ms:
        chunk_end = min(seg.end_ms, cursor + chunk_ms)
        if seg.end_ms - chunk_end < min_segment_ms:
            chunk_end = seg.end_ms
        if seg.pcm16 is not None:
            local_start = cursor - seg.start_ms
            local_end = chunk_end - seg.start_ms
            chunk_pcm = _slice_pcm(seg.pcm16, sample_rate, local_start, local_end)
            out.append(
                Segment(
                    start_ms=cursor,
                    end_ms=chunk_end,
                    wav_path="",
                    pcm16=chunk_pcm,
                    sample_rate=sample_rate,
                )
            )
        else:
            out.append(
                Segment(
                    start_ms=cursor,
                    end_ms=chunk_end,
                    wav_path="",
                    pcm16=None,
                    sample_rate=sample_rate,
                )
            )
        cursor = chunk_end
    return out


def _split_long_segments(
    wav_path: str,
    segments: List[Segment],
    *,
    max_segment_ms: int = 9000,
    min_segment_ms: int = 2500,
) -> List[Segment]:
    """
    Disk-based split: writes split chunks to <wav_dir>/vad_split/. Reads each
    segment's PCM from disk, runs silence-aware splitting, and persists chunks.
    """
    if not segments:
        return []
    out_dir = os.path.join(os.path.dirname(wav_path), "vad_split")
    ensure_dir(out_dir)

    # Re-read each segment's PCM so we can split it.
    enriched: List[Segment] = []
    for seg in segments:
        if seg.pcm16 is None and seg.wav_path and os.path.exists(seg.wav_path):
            pcm, sr = _read_wav_mono16(seg.wav_path)
            enriched.append(
                Segment(
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    wav_path=seg.wav_path,
                    pcm16=pcm,
                    sample_rate=sr,
                )
            )
        else:
            enriched.append(seg)

    use_silence = os.getenv("SPLIT_AT_SILENCE", "1").strip().lower() not in {"0", "false", "no", "off"}
    if use_silence:
        split = _split_long_segments_at_silence(
            enriched,
            max_segment_ms=max_segment_ms,
            min_segment_ms=min_segment_ms,
        )
    else:
        split = []
        for seg in enriched:
            split.extend(_even_split(seg, max_segment_ms, min_segment_ms))

    return _materialize_segments(split, out_dir)


def _denoise_pcm16(pcm: bytes, sample_rate: int) -> bytes:
    """
    Apply spectral gating noise reduction to a mono 16-bit PCM buffer.
    Uses the first 0.5 s of audio as a noise profile. If noisereduce is
    unavailable, returns the input unchanged.
    """
    global _DENOISE_FAILED
    if _DENOISE_FAILED:
        return pcm
    if os.getenv("REFERENCE_NOISE_REDUCE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return pcm
    try:
        import noisereduce as nr  # type: ignore
    except Exception as exc:
        logger.info("noisereduce not available; skipping noise reduction: %s", exc)
        _DENOISE_FAILED = True
        return pcm

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size < sample_rate // 4:
        return pcm
    try:
        # Use the first 0.5 s as a noise profile. For voiced speech this is
        # usually silence or low-energy room tone.
        noise_clip_size = min(audio.size, sample_rate // 2)
        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            y_noise=audio[:noise_clip_size],
            prop_decrease=0.75,
            stationary=False,
        )
        reduced = np.clip(reduced, -1.0, 1.0)
        return (reduced * 32767.0).astype(np.int16).tobytes()
    except Exception as exc:
        logger.warning("noisereduce failed; using original PCM: %s", exc)
        _DENOISE_FAILED = True
        return pcm


def _denoise_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    return _denoise_pcm16(pcm, sample_rate)


def _split_text_for_tts(text: str, *, max_chars: int) -> List[str]:
    compact = " ".join(text.split())
    if not compact:
        return []

    def _chunk_by_words(value: str) -> List[str]:
        words = value.split()
        if not words:
            return []
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for word in words:
            next_len = current_len + len(word) + (1 if current else 0)
            if current and next_len > max_chars:
                chunks.append(" ".join(current).strip())
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len = next_len
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    parts: List[str] = []
    sentences = re.split(r"(?<=[.!?…])\s+|(?<=[;:])\s+", compact)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            parts.append(sentence)
            continue
        clauses = re.split(r"(?<=,)\s+", sentence)
        current = ""
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            candidate = f"{current} {clause}".strip() if current else clause
            if current and len(candidate) > max_chars:
                parts.extend(_chunk_by_words(current))
                current = clause
            else:
                current = candidate
        if current:
            parts.extend(_chunk_by_words(current))
    return [part for part in parts if part]


def _estimate_pitch(audio: np.ndarray, sample_rate: int) -> float:
    if audio.size < sample_rate // 4:
        return 0.0
    audio = audio[: min(audio.size, sample_rate * 2)]
    audio = audio - np.mean(audio)
    if np.max(np.abs(audio)) < 1e-4:
        return 0.0
    audio = audio[::2]
    sr = sample_rate // 2
    corr = np.correlate(audio, audio, mode="full")[audio.size - 1 :]
    min_lag = max(1, int(sr / 350))
    max_lag = max(min_lag + 1, int(sr / 70))
    window = corr[min_lag:max_lag]
    if window.size == 0:
        return 0.0
    lag = int(np.argmax(window)) + min_lag
    if lag <= 0:
        return 0.0
    return float(sr / lag)


def _speaker_embedding_feature(wav_path: str) -> Optional[np.ndarray]:
    global _SPEAKER_ENCODER, _SPEAKER_ENCODER_FAILED

    if _SPEAKER_ENCODER_FAILED:
        return None
    enabled = os.getenv("USE_SPEECHBRAIN_SPEAKER_EMBEDDINGS", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None

    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as exc:
        logger.info("SpeechBrain speaker embeddings unavailable; using basic audio features: %s", exc)
        _SPEAKER_ENCODER_FAILED = True
        return None

    try:
        if _SPEAKER_ENCODER is None:
            model_id = os.getenv("SPEAKER_EMBEDDING_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
            savedir = os.path.join(os.getcwd(), "models", "speaker-embeddings", model_id.replace("/", "_"))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _SPEAKER_ENCODER = EncoderClassifier.from_hparams(
                source=model_id,
                savedir=savedir,
                run_opts={"device": device},
            )
            logger.info("Loaded speaker embedding model: %s device=%s", model_id, device)

        signal = _SPEAKER_ENCODER.load_audio(wav_path)
        embedding = _SPEAKER_ENCODER.encode_batch(signal).squeeze().detach().cpu().numpy()
        embedding = embedding.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-8:
            return None
        return embedding / norm
    except Exception as exc:
        logger.warning("Speaker embedding failed for %s; using basic audio features: %s", wav_path, exc)
        _SPEAKER_ENCODER_FAILED = True
        return None


def _speaker_feature(wav_path: str) -> np.ndarray:
    embedding = _speaker_embedding_feature(wav_path)
    if embedding is not None:
        return embedding

    audio, sample_rate = _read_wav_float(wav_path)
    if audio.size == 0:
        return np.zeros(6, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-8)
    zcr = float(np.mean(audio[:-1] * audio[1:] < 0)) if audio.size > 1 else 0.0
    window = np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(audio * window)) + 1e-8
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / sample_rate)
    centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / np.sum(spectrum)))
    pitch = _estimate_pitch(audio, sample_rate)
    duration = float(audio.size / sample_rate)
    feature = np.array(
        [
            min(rms * 10.0, 2.0),
            min(zcr * 10.0, 2.0),
            min(centroid / 4000.0, 2.0),
            min(bandwidth / 4000.0, 2.0),
            min(pitch / 400.0, 2.0),
            min(duration / 10.0, 2.0),
        ],
        dtype=np.float32,
    )
    return feature


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def _cosine_pairwise(features: List[np.ndarray]) -> np.ndarray:
    n = len(features)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = _cosine_distance(features[i], features[j])
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _agglomerative_cluster(
    features: List[np.ndarray],
    *,
    max_speakers: int,
    distance_threshold: float,
) -> List[int]:
    """
    Agglomerative clustering with average linkage, capped at max_speakers
    clusters. Returns speaker id per row, starting from 0.
    Falls back to a sequential threshold scan if sklearn is unavailable.
    """
    if not features:
        return []
    if len(features) == 1:
        return [0]
    try:
        from sklearn.cluster import AgglomerativeClustering
    except Exception as exc:
        logger.info("sklearn unavailable, falling back to threshold scan: %s", exc)
        return _threshold_cluster(features, max_speakers=max_speakers, threshold=distance_threshold)

    n = len(features)
    if n == 1:
        return [0]
    dist_matrix = _cosine_pairwise(features)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(dist_matrix)
    # Cap cluster count at max_speakers by merging the closest centroids.
    unique_labels = list(dict.fromkeys(int(l) for l in labels))
    while len(unique_labels) > max_speakers:
        # Find two centroids with smallest pairwise distance and merge.
        centroids: Dict[int, np.ndarray] = {}
        counts: Dict[int, int] = {}
        for lbl, feat in zip(labels, features):
            lbl = int(lbl)
            centroids.setdefault(lbl, np.zeros_like(feat))
            centroids[lbl] += feat
            counts[lbl] = counts.get(lbl, 0) + 1
        for lbl in centroids:
            centroids[lbl] /= max(1, counts[lbl])
        # Pairwise distance between centroids
        best_pair: Optional[Tuple[int, int]] = None
        best_dist = float("inf")
        for i_idx, i_lbl in enumerate(unique_labels):
            for j_lbl in unique_labels[i_idx + 1 :]:
                d = _cosine_distance(centroids[i_lbl], centroids[j_lbl])
                if d < best_dist:
                    best_dist = d
                    best_pair = (i_lbl, j_lbl)
        if best_pair is None:
            break
        a, b = best_pair
        merge_into = min(a, b)
        merge_from = max(a, b)
        labels = np.where(labels == merge_from, merge_into, labels)
        unique_labels = list(dict.fromkeys(int(l) for l in labels))
    # Remap labels to dense 0..K-1
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    return [label_map[int(l)] for l in labels]


def _threshold_cluster(
    features: List[np.ndarray],
    *,
    max_speakers: int,
    threshold: float,
) -> List[int]:
    """Original online greedy threshold clustering, kept as a fallback."""
    speaker_centroids: Dict[int, np.ndarray] = {}
    speaker_counts: Dict[int, int] = {}
    assignments: List[int] = []
    for feature in features:
        if not speaker_centroids:
            speaker_centroids[0] = feature.copy()
            speaker_counts[0] = 1
            assignments.append(0)
            continue
        best_id = min(speaker_centroids, key=lambda sid: _cosine_distance(feature, speaker_centroids[sid]))
        best_distance = _cosine_distance(feature, speaker_centroids[best_id])
        if best_distance > threshold and len(speaker_centroids) < max_speakers:
            speaker_id = max(speaker_centroids) + 1
            speaker_centroids[speaker_id] = feature.copy()
            speaker_counts[speaker_id] = 1
            assignments.append(speaker_id)
            continue
        count = speaker_counts[best_id]
        speaker_centroids[best_id] = (speaker_centroids[best_id] * count + feature) / (count + 1)
        speaker_counts[best_id] = count + 1
        assignments.append(best_id)
    return assignments


def _assign_speakers(
    features: List[np.ndarray],
    *,
    max_speakers: int,
    threshold: float = 0.08,
) -> Tuple[List[int], Dict[int, np.ndarray]]:
    """
    Pick clustering strategy: agglomerative by default, online threshold as fallback.
    Returns (assignments, speaker_centroids).
    """
    use_agglo = os.getenv("USE_AGGLOMERATIVE_SPEAKER_CLUSTERING", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }
    if use_agglo:
        assignments = _agglomerative_cluster(
            features, max_speakers=max_speakers, distance_threshold=threshold
        )
    else:
        assignments = _threshold_cluster(
            features, max_speakers=max_speakers, threshold=threshold
        )

    speaker_centroids: Dict[int, np.ndarray] = {}
    speaker_counts: Dict[int, int] = {}
    for sid, feat in zip(assignments, features):
        if sid not in speaker_centroids:
            speaker_centroids[sid] = feat.copy()
            speaker_counts[sid] = 1
        else:
            c = speaker_counts[sid]
            speaker_centroids[sid] = (speaker_centroids[sid] * c + feat) / (c + 1)
            speaker_counts[sid] = c + 1
    return assignments, speaker_centroids


def _effective_speaker_threshold(features: List[np.ndarray], configured: float) -> float:
    if not features:
        return configured
    if features[0].size > 16 and configured <= 0.08:
        return float(os.getenv("SPEECHBRAIN_SPEAKER_THRESHOLD", "0.35"))
    return configured


def _load_profile_store(profile_store_path: Optional[str]) -> List[dict]:
    if not profile_store_path or not os.path.exists(profile_store_path):
        return []
    with open(profile_store_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _pick_reference_segment(
    speaker_id: int,
    segments: List[Segment],
    assignments: List[int],
) -> Optional[Segment]:
    """
    Build a clean reference WAV for this speaker.
    Longer segments → better voice cloning quality.
    """
    speaker_segments = [
        seg for seg, sid in zip(segments, assignments)
        if sid == speaker_id and (seg.end_ms - seg.start_ms) >= 700
    ]
    if not speaker_segments:
        return None

    max_reference_ms = int(os.getenv("VOICE_REFERENCE_MAX_MS", "20000"))
    selected = sorted(
        speaker_segments,
        key=lambda seg: seg.end_ms - seg.start_ms,
        reverse=True,
    )

    combined = AudioSegment.silent(duration=0)
    for seg in selected:
        if len(combined) >= max_reference_ms:
            break
        if seg.wav_path and os.path.exists(seg.wav_path):
            chunk = AudioSegment.from_file(seg.wav_path)
        elif seg.pcm16 is not None:
            chunk = _pcm16_to_audiosegment(seg.pcm16, seg.sample_rate)
        else:
            continue
        remaining = max_reference_ms - len(combined)
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        combined += chunk + AudioSegment.silent(duration=80)

    if not len(combined):
        return selected[0] if selected[0].wav_path else None

    base_path = segments[0].wav_path or "."
    out_dir = os.path.join(os.path.dirname(base_path), "speaker_refs")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"speaker_{speaker_id:02d}_reference.wav")
    combined.export(out_path, format="wav")
    return Segment(start_ms=0, end_ms=len(combined), wav_path=out_path)


def _read_wav_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _save_profile_store(profile_store_path: Optional[str], profiles: List[dict]) -> None:
    if not profile_store_path:
        return
    ensure_dir(os.path.dirname(profile_store_path))
    with open(profile_store_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=True, indent=2)


def _parse_voice_entry(entry: str) -> Tuple[str, str]:
    value = entry.strip()
    if ":" not in value:
        return "any", value
    prefix, voice = value.split(":", 1)
    prefix = prefix.strip().lower()
    voice = voice.strip()
    if prefix in {"boy", "young_male"}:
        return "boy", voice
    if prefix in {"girl", "young_female"}:
        return "girl", voice
    if prefix in {"man", "adult_male"}:
        return "man", voice
    if prefix in {"woman", "adult_female"}:
        return "woman", voice
    if prefix in {"old", "elder", "qari"}:
        return "old", voice
    if prefix in {"m", "male", "boy", "erkak"}:
        return "man", voice
    if prefix in {"f", "female", "girl", "ayol", "qiz"}:
        return "woman", voice
    return "any", voice


def _infer_role_from_centroid(centroid: np.ndarray) -> str:
    if centroid.size < 5:
        return "any"
    estimated_pitch_hz = float(centroid[4]) * 400.0
    if estimated_pitch_hz <= 0.0:
        return "any"
    if estimated_pitch_hz >= 245.0:
        return "girl"
    if estimated_pitch_hz >= 185.0:
        return "woman"
    if estimated_pitch_hz >= 155.0:
        return "boy"
    if estimated_pitch_hz >= 115.0:
        return "man"
    return "old"


def _role_fallbacks(role: str) -> List[str]:
    mapping = {
        "boy": ["boy", "man", "old", "any", "woman", "girl"],
        "girl": ["girl", "woman", "any", "boy", "man", "old"],
        "man": ["man", "boy", "old", "any", "woman", "girl"],
        "woman": ["woman", "girl", "any", "man", "boy", "old"],
        "old": ["old", "man", "woman", "any", "boy", "girl"],
        "any": ["any", "man", "woman", "boy", "girl", "old"],
    }
    return mapping.get(role, mapping["any"])


def _pick_voice(
    *,
    preferred_role: str,
    voice_entries: List[str],
    used_voices: set,
    default_voice: str,
) -> str:
    parsed = [_parse_voice_entry(entry) for entry in voice_entries]
    all_pool = [voice for _gender, voice in parsed if voice]
    for role in _role_fallbacks(preferred_role):
        pool = [voice for category, voice in parsed if category == role and voice]
        voice = next((v for v in pool if v not in used_voices), None)
        if voice:
            return voice
    if default_voice:
        return default_voice
    return all_pool[0] if all_pool else "alloy"


def _select_speaker_voices(
    speaker_centroids: Dict[int, np.ndarray],
    *,
    available_voices: List[str],
    default_voice: str,
    profile_store_path: Optional[str],
    match_threshold: float = 0.06,
    speaker_reference_segments: Optional[Dict[int, Segment]] = None,
) -> Tuple[Dict[int, str], Dict[int, Optional[str]]]:
    """
    Returns:
      speaker_voices  : {speaker_id -> voice_name_or_id}
      speaker_ref_paths : {speaker_id -> reference_wav_path_or_None}
    """
    profiles = _load_profile_store(profile_store_path)
    used_profile_indexes: set = set()
    allowed_voice_entries = [voice.strip() for voice in available_voices if voice.strip()]
    allowed_voices = {_parse_voice_entry(voice)[1] for voice in allowed_voice_entries}
    used_voices = {
        str(p.get("voice", "")).strip()
        for p in profiles
        if str(p.get("voice", "")).strip() in allowed_voices
    }
    speaker_voices: Dict[int, str] = {}
    speaker_ref_paths: Dict[int, Optional[str]] = {}

    for speaker_id, centroid in speaker_centroids.items():
        best_index = None
        best_distance = 99.0
        for idx, profile in enumerate(profiles):
            if idx in used_profile_indexes:
                continue
            saved = np.array(profile.get("centroid", []), dtype=np.float32)
            if saved.size != centroid.size:
                continue
            distance = _cosine_distance(centroid, saved)
            if distance < best_distance:
                best_distance = distance
                best_index = idx

        ref_path: Optional[str] = None
        ref_seg = (speaker_reference_segments or {}).get(speaker_id)
        if ref_seg:
            ref_path = ref_seg.wav_path

        if best_index is not None and best_distance <= match_threshold:
            used_profile_indexes.add(best_index)
            voice = str(profiles[best_index].get("voice", "")).strip()
            preferred_role = (
                str(profiles[best_index].get("role", "")).strip()
                or str(profiles[best_index].get("gender", "")).strip()
                or _infer_role_from_centroid(centroid)
            )
            if voice not in allowed_voices:
                voice = _pick_voice(
                    preferred_role=preferred_role,
                    voice_entries=allowed_voice_entries,
                    used_voices=used_voices,
                    default_voice=default_voice,
                )
                profiles[best_index]["voice"] = voice
            profiles[best_index]["role"] = preferred_role
            profiles[best_index]["centroid"] = centroid.tolist()
            saved_ref = profiles[best_index].get("reference_wav")
            if ref_path:
                profiles[best_index]["reference_wav"] = ref_path
            elif saved_ref and os.path.exists(saved_ref):
                ref_path = saved_ref
            speaker_voices[speaker_id] = voice
            speaker_ref_paths[speaker_id] = ref_path
            used_voices.add(voice)
            continue

        inferred_role = _infer_role_from_centroid(centroid)
        voice = _pick_voice(
            preferred_role=inferred_role,
            voice_entries=allowed_voice_entries,
            used_voices=used_voices,
            default_voice=default_voice,
        )
        new_profile: dict = {
            "voice": voice,
            "role": inferred_role,
            "centroid": centroid.tolist(),
        }
        if ref_path:
            new_profile["reference_wav"] = ref_path
        profiles.append(new_profile)
        used_voices.add(voice)
        speaker_voices[speaker_id] = voice
        speaker_ref_paths[speaker_id] = ref_path

    _save_profile_store(profile_store_path, profiles[-64:])
    return speaker_voices, speaker_ref_paths


def _demucs_two_stems(input_wav: str, out_dir: str) -> Tuple[str, str]:
    ensure_dir(out_dir)
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        "htdemucs",
        "--two-stems=vocals",
        "--out",
        out_dir,
        input_wav,
    ]
    run(cmd)
    base = os.path.splitext(os.path.basename(input_wav))[0]
    vocals = os.path.join(out_dir, "htdemucs", base, "vocals.wav")
    no_vocals = os.path.join(out_dir, "htdemucs", base, "no_vocals.wav")
    if not (os.path.exists(vocals) and os.path.exists(no_vocals)):
        raise RuntimeError("Demucs output not found")
    return vocals, no_vocals


def _atempo_chain(playback_ratio: float) -> str:
    playback_ratio = max(0.25, min(playback_ratio, 4.0))
    parts: List[str] = []
    while playback_ratio > 2.0:
        parts.append("atempo=2.0")
        playback_ratio /= 2.0
    while playback_ratio < 0.5:
        parts.append("atempo=0.5")
        playback_ratio /= 0.5
    parts.append(f"atempo={playback_ratio:.5f}")
    return ",".join(parts)


def _fit_tts_segment(
    *,
    ffmpeg: str,
    input_path: str,
    output_path: str,
    target_ms: int,
    min_playback_ratio: float,
    max_playback_ratio: float,
    target_dbfs: Optional[float] = None,
) -> None:
    source_ms = max(1, len(AudioSegment.from_file(input_path)))
    playback_ratio = max(min_playback_ratio, min(source_ms / max(1, target_ms), max_playback_ratio))
    audio_filter = f"{_atempo_chain(playback_ratio)}"
    run([ffmpeg, "-y", "-i", input_path, "-af", audio_filter, output_path])
    fitted = AudioSegment.from_file(output_path)

    if target_dbfs is not None and fitted.dBFS != float('-inf'):
        change_in_dbfs = target_dbfs - fitted.dBFS
        fitted = fitted.apply_gain(change_in_dbfs)

    if len(fitted) > target_ms:
        fitted = fitted[:target_ms]
    elif len(fitted) < target_ms:
        fitted += AudioSegment.silent(duration=target_ms - len(fitted))
    fitted.export(output_path, format="wav")


def _render_tts_segment(
    *,
    client: SpeechClient,
    ffmpeg: str,
    tts_dir: str,
    segment_key: str,
    text: str,
    voice: str,
    target_ms: int,
    split_max_chars: int,
    chunk_pause_ms: int,
    min_playback_ratio: float,
    max_playback_ratio: float,
    emotion_instructions: Optional[str] = None,
    emotion_speed: Optional[float] = None,
    emotion: Optional[str] = None,
    reference_audio: Optional[bytes] = None,
    reference_text: str = "",
    target_dbfs: Optional[float] = None,
    extra_references: Optional[list] = None,
) -> AudioSegment:
    chunks = _split_text_for_tts(text, max_chars=split_max_chars)
    if not chunks:
        return AudioSegment.silent(duration=target_ms)

    total_chars = sum(max(1, len(chunk)) for chunk in chunks)
    total_pause_ms = max(0, len(chunks) - 1) * chunk_pause_ms
    available_ms = max(800, target_ms - total_pause_ms)
    rendered = AudioSegment.silent(duration=0)
    remaining_ms = available_ms
    remaining_chars = total_chars

    for idx, chunk in enumerate(chunks):
        chunk_chars = max(1, len(chunk))
        remaining_chunks = len(chunks) - idx - 1
        if idx == len(chunks) - 1:
            chunk_target_ms = max(600, remaining_ms)
        else:
            proportional_ms = int(remaining_ms * (chunk_chars / max(1, remaining_chars)))
            min_reserved_ms = remaining_chunks * 600
            chunk_target_ms = max(600, min(proportional_ms, max(600, remaining_ms - min_reserved_ms)))

        raw_path = os.path.join(tts_dir, f"{segment_key}_part_{idx:02d}_raw.wav")
        norm_path = os.path.join(tts_dir, f"{segment_key}_part_{idx:02d}.wav")

        tts_kwargs = {
            "voice": voice,
            "instructions": emotion_instructions,
            "speed": emotion_speed,
            "emotion": emotion,
            "reference_audio": reference_audio,
            "reference_text": reference_text,
        }
        if extra_references and getattr(client, "__class__", None).__name__ == "LocalSpeechClient":
            tts_kwargs["extra_references"] = extra_references

        audio_bytes = client.tts(chunk, **tts_kwargs)

        with open(raw_path, "wb") as f:
            f.write(audio_bytes)
        _fit_tts_segment(
            ffmpeg=ffmpeg,
            input_path=raw_path,
            output_path=norm_path,
            target_ms=chunk_target_ms,
            min_playback_ratio=min_playback_ratio,
            max_playback_ratio=max_playback_ratio,
            target_dbfs=target_dbfs,
        )
        chunk_audio = AudioSegment.from_file(norm_path)
        rendered += chunk_audio
        if idx < len(chunks) - 1:
            rendered += AudioSegment.silent(duration=chunk_pause_ms)

        remaining_ms = max(0, remaining_ms - chunk_target_ms)
        remaining_chars = max(1, remaining_chars - chunk_chars)

    if len(rendered) > target_ms:
        rendered = rendered[:target_ms]
    elif len(rendered) < target_ms:
        rendered += AudioSegment.silent(duration=target_ms - len(rendered))
    return rendered


def _loudness_normalize(audio: AudioSegment, target_lufs: float) -> AudioSegment:
    """
    Normalize a mono/stereo AudioSegment to target_lufs (typically -16 LUFS
    for broadcast dialog). Falls back to no-op if pyloudnorm is unavailable.
    """
    global _LOUDNORM_FAILED
    if _LOUDNORM_FAILED:
        return audio
    if os.getenv("DIALOG_LOUDNESS_NORMALIZE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return audio
    try:
        import pyloudnorm as pyln  # type: ignore
    except Exception as exc:
        logger.info("pyloudnorm not available; skipping loudness normalization: %s", exc)
        _LOUDNORM_FAILED = True
        return audio

    try:
        sample_width = audio.sample_width
        channels = audio.channels
        sample_rate = audio.frame_rate
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if channels > 1:
            samples = samples.reshape((-1, channels))
        samples /= max(1.0, float(2 ** (8 * sample_width - 1)))
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(samples)
        if not np.isfinite(loudness):
            return audio
        normalized = pyln.normalize.loudness(samples, loudness, target_lufs)
        peak = float(np.max(np.abs(normalized)))
        if peak > 0.99:
            normalized = normalized * (0.99 / peak)
        pcm16 = (np.clip(normalized, -1.0, 1.0) * 32767.0).astype(np.int16)
        if channels > 1:
            pcm16 = pcm16.reshape(-1)
        return AudioSegment(pcm16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=channels)
    except Exception as exc:
        logger.warning("Loudness normalization failed; leaving audio unchanged: %s", exc)
        _LOUDNORM_FAILED = True
        return audio


# ── Prosody, emotion, and quality helpers ─────────────────────────────────────

def _compute_audio_prosody(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Extract a dictionary of prosodic features from a mono float32 audio buffer:
      rms_mean, rms_std, zcr, pitch_mean_hz, pitch_std_hz,
      voiced_ratio, spectral_centroid, duration_s, snr_db
    """
    out = {
        "rms_mean": 0.0,
        "rms_std": 0.0,
        "zcr": 0.0,
        "pitch_mean_hz": 0.0,
        "pitch_std_hz": 0.0,
        "voiced_ratio": 0.0,
        "spectral_centroid": 0.0,
        "duration_s": float(audio.size / max(1, sample_rate)),
        "snr_db": 0.0,
    }
    if audio.size < sample_rate // 8:
        return out
    try:
        # Frame-level analysis (40 ms frames, 50% overlap)
        frame_size = int(sample_rate * 0.04)
        hop = max(1, frame_size // 2)
        if frame_size < 8:
            return out
        rms_vals: List[float] = []
        zcr_vals: List[float] = []
        voiced_flags: List[bool] = []
        centroid_vals: List[float] = []
        for start in range(0, audio.size - frame_size, hop):
            frame = audio[start:start + frame_size]
            r = float(np.sqrt(np.mean(frame * frame)) + 1e-8)
            rms_vals.append(r)
            zcr_vals.append(float(np.mean(frame[:-1] * frame[1:] < 0)) if frame.size > 1 else 0.0)
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame.size))) + 1e-8
            freqs = np.fft.rfftfreq(frame.size, d=1.0 / sample_rate)
            centroid_vals.append(float(np.sum(freqs * spectrum) / np.sum(spectrum)))
            # Simple voicing heuristic: pitch estimate > 60 Hz and RMS above noise floor
            pitch = _estimate_pitch(frame, sample_rate)
            voiced_flags.append(pitch > 60.0 and r > 0.01)
        if rms_vals:
            rms_arr = np.array(rms_vals)
            out["rms_mean"] = float(np.mean(rms_arr))
            out["rms_std"] = float(np.std(rms_arr))
            out["zcr"] = float(np.mean(zcr_vals)) if zcr_vals else 0.0
            out["voiced_ratio"] = float(np.mean(voiced_flags)) if voiced_flags else 0.0
            out["spectral_centroid"] = float(np.mean(centroid_vals)) if centroid_vals else 0.0
        # Pitch over the whole buffer (single estimate, fast)
        pitch_full = _estimate_pitch(audio, sample_rate)
        out["pitch_mean_hz"] = float(pitch_full)
        # Approximate pitch std by chunking the buffer into 4 windows
        chunk = max(sample_rate, audio.size // 4)
        chunk_pitches: List[float] = []
        for off in range(0, audio.size, chunk):
            sub = audio[off:off + chunk]
            if sub.size < sample_rate // 4:
                break
            p = _estimate_pitch(sub, sample_rate)
            if p > 50.0:
                chunk_pitches.append(p)
        if chunk_pitches:
            out["pitch_std_hz"] = float(np.std(chunk_pitches))
        # Crude SNR estimate: ratio of top-decile RMS to bottom-decile RMS
        if rms_vals:
            sorted_rms = np.sort(np.array(rms_vals))
            top = float(np.mean(sorted_rms[-max(1, len(sorted_rms) // 10):]))
            bot = float(np.mean(sorted_rms[: max(1, len(sorted_rms) // 10)]) + 1e-8)
            out["snr_db"] = float(20.0 * np.log10(top / bot))
    except Exception as exc:
        logger.debug("Prosody extraction failed: %s", exc)
    return out


def _detect_emotion_from_features(prosody: dict) -> str:
    """
    Map a prosody feature dict to a Fish Speech emotion tag.
    Heuristics tuned for general speech; not a clinical classifier.
    Returns one of: "gazab", "yiglash", "qaygu", "qorquv",
                    "xursandchilik", "hayrat", "sovuqqonlik",
                    "qahramonona", "yovuz_qahramon", "any".
    """
    rms = prosody.get("rms_mean", 0.0)
    rms_std = prosody.get("rms_std", 0.0)
    zcr = prosody.get("zcr", 0.0)
    pitch = prosody.get("pitch_mean_hz", 0.0)
    pitch_std = prosody.get("pitch_std_hz", 0.0)
    centroid = prosody.get("spectral_centroid", 0.0)
    voiced = prosody.get("voiced_ratio", 0.0)

    if voiced < 0.1:
        return "any"

    # Anger / shouting: loud + high pitch variance + bright spectrum
    if rms > 0.08 and pitch_std > 25 and centroid > 1500:
        return "gazab"
    if rms > 0.12:
        return "gazab"

    # Happy / excited: high pitch + high ZCR + bright
    if pitch > 180 and zcr > 0.08 and rms_std > 0.02:
        return "xursandchilik"

    # Sad: low energy + low pitch + monotone
    if rms < 0.03 and pitch_std < 15 and voiced > 0.2:
        return "qaygu"

    # Sobbing/crying: low energy, very low pitch, voiced but weak
    if rms < 0.025 and pitch < 140 and pitch_std < 12 and voiced > 0.15:
        return "yiglash"

    # Fear / scared: high pitch, low RMS, high ZCR
    if pitch > 200 and rms < 0.05 and zcr > 0.10:
        return "qorquv"

    # Surprise: very high pitch jump
    if pitch > 220 and pitch_std > 30:
        return "hayrat"

    # Heroic / confident: stable medium-high pitch, strong energy
    if 150 < pitch < 200 and 0.04 < rms < 0.10 and pitch_std < 18:
        return "qahramonona"

    # Villainous: low pitch, low centroid (dark timbre)
    if pitch < 130 and centroid < 1200 and rms < 0.06:
        return "yovuz_qahramon"

    # Calm / neutral: moderate everything, low std
    if pitch_std < 12 and rms_std < 0.015:
        return "sovuqqonlik"

    return "any"


def _detect_emotion_from_wav(wav_path: str) -> Tuple[str, dict]:
    """
    Convenience wrapper. Returns (emotion_tag, prosody_dict).
    """
    try:
        audio, sr = _read_wav_float(wav_path)
        prosody = _compute_audio_prosody(audio, sr)
        return _detect_emotion_from_features(prosody), prosody
    except Exception as exc:
        logger.debug("Emotion detection failed for %s: %s", wav_path, exc)
        return "any", {}


def _validate_reference_audio(wav_path: str) -> Tuple[bool, dict]:
    """
    Check whether a reference audio is good enough for voice cloning.
    Returns (is_valid, prosody_dict). Logs the reason if invalid.
    """
    try:
        audio, sr = _read_wav_float(wav_path)
        duration_s = audio.size / max(1, sr)
        prosody = _compute_audio_prosody(audio, sr)
        min_seconds = float(os.getenv("REFERENCE_MIN_SECONDS", "3.0"))
        min_voiced = float(os.getenv("REFERENCE_MIN_VOICED_RATIO", "0.3"))
        min_snr = float(os.getenv("REFERENCE_MIN_SNR_DB", "6.0"))
        if duration_s < min_seconds:
            logger.info("Reference too short (%.2fs < %.2fs): %s", duration_s, min_seconds, wav_path)
            return False, prosody
        if prosody.get("voiced_ratio", 0.0) < min_voiced:
            logger.info("Reference voiced ratio too low (%.2f < %.2f): %s",
                        prosody["voiced_ratio"], min_voiced, wav_path)
            return False, prosody
        if prosody.get("snr_db", 0.0) < min_snr:
            logger.info("Reference SNR too low (%.1f dB < %.1f dB): %s",
                        prosody["snr_db"], min_snr, wav_path)
            return False, prosody
        return True, prosody
    except Exception as exc:
        logger.warning("Reference validation failed for %s: %s", wav_path, exc)
        return False, {}


def _validate_tts_output(wav_bytes: bytes) -> bool:
    """
    Reject obviously broken TTS output (silent, clipped, or near-empty).
    Returns True if the output looks usable.
    """
    if not wav_bytes or len(wav_bytes) < 1024:
        return False
    try:
        import io
        with contextlib.closing(wave.open(io.BytesIO(wav_bytes), "rb")) as wf:
            if wf.getnchannels() not in (1, 2) or wf.getsampwidth() != 2:
                return False
            sr = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())
        if not pcm:
            return False
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return False
        duration_s = audio.size / max(1, sr)
        if duration_s < 0.2:
            return False
        rms = float(np.sqrt(np.mean(audio * audio)))
        if rms < 0.005:
            logger.warning("TTS output near-silent (rms=%.4f)", rms)
            return False
        # Clipping detection: > 5% samples at the extremes
        clip_ratio = float(np.mean(np.abs(audio) > 0.99))
        if clip_ratio > 0.05:
            logger.warning("TTS output heavily clipped (%.1f%% samples)", clip_ratio * 100)
            return False
        return True
    except Exception as exc:
        logger.warning("TTS validation error: %s", exc)
        return False


def _estimate_speaking_rate(words: List[dict]) -> Tuple[float, List[float]]:
    """
    Given Whisper word timestamps, return (syllables_per_sec, inter_word_pauses_ms).
    Syllables are approximated as max(1, len(word)/3).
    """
    if not words or len(words) < 2:
        return 0.0, []
    total_ms = max(1, words[-1]["end_ms"] - words[0]["start_ms"])
    total_syllables = sum(max(1, len(w["word"].strip()) // 3) for w in words)
    rate = float(total_syllables) / (total_ms / 1000.0)
    pauses: List[float] = []
    for prev, curr in zip(words, words[1:]):
        gap = curr["start_ms"] - prev["end_ms"]
        if gap > 0:
            pauses.append(float(gap))
    return rate, pauses


def _select_best_reference_segments(
    speaker_segments: List[Segment],
    *,
    max_total_ms: int = 20000,
    max_segments: int = 4,
) -> List[Segment]:
    """
    Pick the top-N best reference segments for a speaker.
    Score = length * voiced_ratio * (1 + snr/20).
    Returns at most max_segments, total duration ≤ max_total_ms.
    """
    if not speaker_segments:
        return []
    scored: List[Tuple[float, Segment]] = []
    for seg in speaker_segments:
        if seg.end_ms - seg.start_ms < 700:
            continue
        path = seg.wav_path
        if not path or not os.path.exists(path):
            continue
        try:
            audio, sr = _read_wav_float(path)
            prosody = _compute_audio_prosody(audio, sr)
        except Exception:
            continue
        length = seg.end_ms - seg.start_ms
        score = (
            float(length)
            * max(0.05, prosody.get("voiced_ratio", 0.2))
            * (1.0 + max(0.0, prosody.get("snr_db", 6.0)) / 20.0)
        )
        scored.append((score, seg))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen: List[Segment] = []
    total_ms = 0
    for _score, seg in scored:
        if len(chosen) >= max_segments:
            break
        seg_len = seg.end_ms - seg.start_ms
        if total_ms + seg_len > max_total_ms and chosen:
            continue
        chosen.append(seg)
        total_ms += seg_len
    return chosen


def _build_combined_reference(
    ref_segments: List[Segment],
    speaker_id: int,
    out_dir: str,
) -> Optional[Segment]:
    """
    Combine the top reference segments into a single clean WAV for cloning.
    Returns a Segment pointing at the new file, or None if nothing usable.
    """
    if not ref_segments:
        return None
    ensure_dir(out_dir)
    combined = AudioSegment.silent(duration=0)
    for seg in ref_segments:
        if seg.wav_path and os.path.exists(seg.wav_path):
            chunk = AudioSegment.from_file(seg.wav_path)
        elif seg.pcm16 is not None:
            chunk = _pcm16_to_audiosegment(seg.pcm16, seg.sample_rate)
        else:
            continue
        combined += chunk + AudioSegment.silent(duration=80)
    if not len(combined):
        return None
    out_path = os.path.join(out_dir, f"speaker_{speaker_id:02d}_reference.wav")
    combined.export(out_path, format="wav")
    return Segment(start_ms=0, end_ms=len(combined), wav_path=out_path)


def _per_word_tempo_text(
    uz_text: str,
    words: List[dict],
    *,
    total_target_ms: int,
) -> str:
    """
    Build a TTS prompt that reflects the original speaking rhythm.
    Strategy: split uz_text into roughly equal chunks, one per Whisper word,
    so each TTS chunk corresponds to one original word. This keeps the
    natural prosodic grouping while letting atempo fix the fine timing.
    """
    n_words = max(1, len(words))
    uz_words = uz_text.split()
    if not uz_words or n_words == 0:
        return uz_text
    # Map each Uzbek word to one or more source-word durations.
    # Simple equal allocation; if counts differ we average.
    ratio = len(uz_words) / float(n_words)
    grouped: List[str] = []
    idx = 0.0
    for i in range(n_words):
        end = idx + ratio
        j_end = int(round(end))
        j_end = min(max(j_end, int(idx) + 1 if i < n_words - 1 else int(idx)), len(uz_words))
        if j_end <= int(round(idx)):
            j_end = min(len(uz_words), int(round(idx)) + 1)
        start_i = int(round(idx))
        chunk = " ".join(uz_words[start_i:j_end]).strip()
        if chunk:
            grouped.append(chunk)
        idx = float(j_end)
        if int(round(idx)) >= len(uz_words):
            break
    if not grouped:
        return uz_text
    return " ".join(grouped)


def _inject_tempo_pauses(
    tts_audio: AudioSegment,
    original_words: List[dict],
    segment_start_ms: int,
    segment_end_ms: int,
) -> AudioSegment:
    """
    Adjust the rendered TTS audio so the relative inter-word pauses roughly
    match the original. This is a coarse approximation: we scale the audio
    uniformly and then insert proportional silences at the major pause points.
    """
    if not original_words or len(tts_audio) < 200:
        return tts_audio
    # Compute the major pauses in the original (gaps > 250 ms)
    major_pauses: List[int] = []  # gaps in ms, sorted desc
    for prev, curr in zip(original_words, original_words[1:]):
        gap = curr["start_ms"] - prev["end_ms"]
        if gap > 250:
            major_pauses.append(int(gap))
    major_pauses.sort(reverse=True)
    if not major_pauses:
        return tts_audio
    # Keep only the top N pauses to avoid over-padding
    keep = min(3, len(major_pauses))
    extra_silence = sum(major_pauses[:keep]) // 2  # 50% of original gap
    extra_silence = min(extra_silence, 600)
    if extra_silence > 50:
        tts_audio = tts_audio + AudioSegment.silent(duration=extra_silence)
    return tts_audio


def _transcribe_segments_parallel(
    client: SpeechClient,
    segments: List[Segment],
    *,
    workers: int,
) -> List[str]:
    """
    Transcribe all segments in parallel. Each segment needs its wav_path
    to exist on disk; the caller is responsible for materialization.
    """
    if not segments:
        return []

    def _one(seg: Segment) -> str:
        if not seg.wav_path or not os.path.exists(seg.wav_path):
            return ""
        return client.transcribe(seg.wav_path) or ""

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(_one, segments))


def dub_media(
    *,
    client: SpeechClient,
    input_path: str,
    work_dir: str,
    has_video: bool,
    available_voices: Optional[List[str]] = None,
    default_voice: Optional[str] = None,
    max_speakers: int = 6,
    profile_store_path: Optional[str] = None,
    target_lang_hint: str = "uz",
    vad_aggressiveness: int = 3,
    max_segment_ms: int = 6500,
    min_segment_ms: int = 1800,
    speaker_match_threshold: float = 0.055,
    tts_min_playback_ratio: float = 0.92,
    tts_max_playback_ratio: float = 1.18,
    tts_split_max_chars: int = 220,
    tts_chunk_pause_ms: int = 140,
    emotion: Optional[str] = None,
) -> Tuple[str, str]:
    ffmpeg = require_ffmpeg()
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(work_dir, job_id)
    ensure_dir(job_dir)
    logger.info("Dub job started: job_id=%s has_video=%s input=%s", job_id, has_video, input_path)

    # Load emotion settings
    emotion_instructions = None
    emotion_speed = None
    emotions_path = os.path.join(os.path.dirname(__file__), "emotions.json")
    if emotion and os.path.exists(emotions_path):
        try:
            with open(emotions_path, "r", encoding="utf-8") as f:
                emotions_data = json.load(f)
            emotions = emotions_data.get("emotions", {})
            if emotion in emotions:
                emotion_instructions = emotions[emotion].get("instructions")
                emotion_speed = emotions[emotion].get("speed")
                logger.info("Using emotion: %s", emotion)
        except Exception as e:
            logger.warning("Failed to load emotions: %s", e)

    audio_for_demucs = os.path.join(job_dir, "src_audio.wav")
    logger.info("Extracting source audio for job_id=%s", job_id)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            audio_for_demucs,
        ]
    )

    stems_dir = os.path.join(job_dir, "stems")
    logger.info("Running Demucs for job_id=%s", job_id)
    vocals_wav, bg_wav = _demucs_two_stems(audio_for_demucs, stems_dir)

    vocals_16k = os.path.join(job_dir, "vocals_16k.wav")
    logger.info("Preparing mono 16k vocals for VAD: job_id=%s", job_id)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            vocals_wav,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            vocals_16k,
        ]
    )

    # ── VAD: in-memory first, then materialize to disk once ────────────────
    pcm_16k, sr_16k = _read_wav_mono16(vocals_16k)
    raw_segments = _vad_segments_in_memory(
        pcm_16k, sr_16k, aggressiveness=vad_aggressiveness, frame_ms=30
    )
    # Split long segments with silence awareness (still in memory)
    use_silence = os.getenv("SPLIT_AT_SILENCE", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }
    if use_silence:
        split_segments = _split_long_segments_at_silence(
            raw_segments,
            max_segment_ms=max_segment_ms,
            min_segment_ms=min_segment_ms,
        )
    else:
        split_segments = []
        for seg in raw_segments:
            split_segments.extend(_even_split(seg, max_segment_ms, min_segment_ms))
    if not split_segments:
        raise RuntimeError("No speech segments detected")
    # Persist each segment exactly once
    segments = _materialize_segments(split_segments, os.path.join(job_dir, "vad_split"))
    logger.info("VAD finished: job_id=%s segments=%s", job_id, len(segments))

    features = [_speaker_feature(seg.wav_path) for seg in segments]
    effective_speaker_match_threshold = _effective_speaker_threshold(
        features, speaker_match_threshold
    )
    speaker_assignments, speaker_centroids = _assign_speakers(
        features,
        max_speakers=max_speakers,
        threshold=effective_speaker_match_threshold,
    )
    logger.info(
        "Speaker clustering finished: job_id=%s speakers=%s",
        job_id, len(speaker_centroids),
    )

    # ── Multi-reference selection: pick 2-4 best segments per speaker ─────
    speaker_best_segments: Dict[int, List[Segment]] = {}
    for sid in speaker_centroids:
        speaker_segs = [
            seg for seg, sid_seg in zip(segments, speaker_assignments)
            if sid_seg == sid and (seg.end_ms - seg.start_ms) >= 700
        ]
        speaker_best_segments[sid] = _select_best_reference_segments(
            speaker_segs,
            max_total_ms=int(os.getenv("VOICE_REFERENCE_MAX_MS", "20000")),
            max_segments=int(os.getenv("REFERENCE_MAX_SEGMENTS", "4")),
        )

    speaker_refs_dir = os.path.join(job_dir, "speaker_refs")
    speaker_combined_refs: Dict[int, Optional[Segment]] = {
        sid: _build_combined_reference(segs, sid, speaker_refs_dir)
        for sid, segs in speaker_best_segments.items()
    }

    # Backward-compat: produce a single Segment per speaker (longest one) for
    # the profile store and the legacy speaker_reference_segments path.
    speaker_reference_segments: Dict[int, Optional[Segment]] = {}
    for sid, segs in speaker_best_segments.items():
        if not segs:
            speaker_reference_segments[sid] = None
        else:
            # Reuse the combined reference (or fall back to the longest)
            if speaker_combined_refs.get(sid) is not None:
                speaker_reference_segments[sid] = speaker_combined_refs[sid]
            else:
                speaker_reference_segments[sid] = max(segs, key=lambda s: s.end_ms - s.start_ms)

    speaker_voices, speaker_ref_paths = _select_speaker_voices(
        speaker_centroids,
        available_voices=available_voices or [default_voice or "alloy"],
        default_voice=default_voice or "alloy",
        profile_store_path=profile_store_path,
        speaker_reference_segments={k: v for k, v in speaker_reference_segments.items() if v},
    )

    tts_dir = os.path.join(job_dir, "tts")
    ensure_dir(tts_dir)
    bg_duration_ms = int(AudioSegment.from_file(bg_wav).duration_seconds * 1000)
    dialog = AudioSegment.silent(duration=bg_duration_ms)

    # ── Step 1: ASR for all segments (parallel) ─────────────────────────────
    tts_workers = max(1, int(os.getenv("TTS_PARALLEL_WORKERS", "2")))
    logger.info("Transcribing %s segments in parallel (workers=%s)", len(segments), tts_workers)
    seg_texts = _transcribe_segments_parallel(client, segments, workers=tts_workers)

    # ── Step 2: batch translation ──────────────────────────────────────────
    # Detect a single source language for the whole job from the first non-empty
    # transcription; the heuristic lives in the local translator.
    translate_batch_size = int(os.getenv("TRANSLATE_BATCH_SIZE", "8"))
    src_lang_hint: Optional[str] = None
    detect_lang = getattr(client, "_get_translator", None)
    if callable(detect_lang):
        try:
            first_text = next((t for t in seg_texts if t.strip()), "")
            if first_text:
                translator = detect_lang()
                src_lang_hint = translator._detect_lang(first_text)
        except Exception:
            src_lang_hint = None

    translations: List[str] = []
    if hasattr(client, "translate_batch_to_uzbek"):
        for start in range(0, len(seg_texts), translate_batch_size):
            batch = seg_texts[start:start + translate_batch_size]
            try:
                out = client.translate_batch_to_uzbek(batch)
            except Exception as exc:
                logger.warning("Batch translation failed (%s); falling back to per-item", exc)
                out = [client.translate_to_uzbek(t) for t in batch]
            # Pad to batch length in case of mismatch
            if len(out) < len(batch):
                out = out + [""] * (len(batch) - len(out))
            translations.extend(out[:len(batch)])
    else:
        translations = [client.translate_to_uzbek(t) for t in seg_texts]
    logger.info("Translation finished: job_id=%s items=%s", job_id, len(translations))

    # ── Step 2b: per-segment emotion + word-level timing ───────────────────
    auto_emotion = os.getenv("AUTO_DETECT_EMOTION", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }
    fetch_word_timestamps = os.getenv("USE_WORD_TIMESTAMPS", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }
    seg_emotions: List[str] = []
    seg_words: List[List[dict]] = []
    if auto_emotion or fetch_word_timestamps:
        for i, seg in enumerate(segments):
            emo = emotion or "any"
            words: List[dict] = []
            if auto_emotion and not emotion:
                emo_label, _prosody = _detect_emotion_from_wav(seg.wav_path)
                if emo_label and emo_label != "any":
                    emo = emo_label
            if fetch_word_timestamps and seg.wav_path and os.path.exists(seg.wav_path):
                try:
                    transcribe_words = getattr(client, "transcribe_with_word_timestamps", None)
                    if callable(transcribe_words):
                        words = transcribe_words(seg.wav_path) or []
                except Exception as exc:
                    logger.debug("Word timestamps failed for seg %s: %s", i, exc)
                    words = []
            seg_emotions.append(emo)
            seg_words.append(words)
        if auto_emotion:
            from collections import Counter
            counts = Counter(seg_emotions)
            logger.info("Auto-detected emotions: %s", dict(counts))
    else:
        seg_emotions = [emotion or "any"] * len(segments)
        seg_words = [[] for _ in segments]

    # ── Step 3: prepare denoised reference audio per speaker ───────────────
    speaker_ref_audio: Dict[int, Optional[bytes]] = {}
    # Reuse the longest ASR text per speaker as reference text — avoids
    # an extra transcribe() call on the reference audio.
    speaker_ref_text: Dict[int, str] = {}
    for sid, ref_wav_path in speaker_ref_paths.items():
        if ref_wav_path and os.path.exists(ref_wav_path):
            try:
                pcm = _read_wav_bytes(ref_wav_path)
                if os.getenv("REFERENCE_NOISE_REDUCE", "1").strip().lower() not in {
                    "0", "false", "no", "off"
                }:
                    with contextlib.closing(wave.open(ref_wav_path, "rb")) as wf:
                        ref_sr = wf.getframerate()
                    pcm = _denoise_pcm16(pcm, ref_sr)
                speaker_ref_audio[sid] = pcm
                logger.info("Loaded reference audio for speaker=%s (%s)", sid, ref_wav_path)
            except Exception as load_err:
                logger.warning("Could not load reference audio for speaker=%s: %s", sid, load_err)
                speaker_ref_audio[sid] = None
        else:
            speaker_ref_audio[sid] = None
        # Pick the longest transcription belonging to this speaker as the
        # reference transcript for voice cloning.
        candidate_texts = sorted(
            (seg_texts[i] for i, sid_seg in enumerate(speaker_assignments) if sid_seg == sid and seg_texts[i].strip()),
            key=len,
            reverse=True,
        )
        speaker_ref_text[sid] = candidate_texts[0] if candidate_texts else ""

    # ── Step 4: TTS + overlay (parallel) ───────────────────────────────────
    def _process_one(i: int, seg: Segment) -> Tuple[int, AudioSegment]:
        speaker_id = speaker_assignments[i]
        voice = speaker_voices.get(speaker_id, default_voice or "alloy")
        ref_audio = speaker_ref_audio.get(speaker_id)
        ref_text = speaker_ref_text.get(speaker_id, "")
        seg_text = seg_texts[i]
        uz_text = translations[i]
        seg_emo = seg_emotions[i] if i < len(seg_emotions) else (emotion or "any")
        words = seg_words[i] if i < len(seg_words) else []
        logger.info(
            "Processing segment: job_id=%s index=%s speaker=%s voice=%s has_ref=%s emotion=%s start_ms=%s end_ms=%s",
            job_id, i, speaker_id, voice, bool(ref_audio), seg_emo, seg.start_ms, seg.end_ms,
        )
        if not uz_text.strip():
            logger.info("Skipped empty translated segment: job_id=%s index=%s", job_id, i)
            return i, AudioSegment.silent(duration=0)
        max_len_ms = max(200, seg.end_ms - seg.start_ms)

        if seg.wav_path and os.path.exists(seg.wav_path):
            orig_audio_seg = AudioSegment.from_file(seg.wav_path)
        elif seg.pcm16 is not None:
            orig_audio_seg = _pcm16_to_audiosegment(seg.pcm16, seg.sample_rate)
        else:
            orig_audio_seg = None
        orig_dbfs = orig_audio_seg.dBFS if (orig_audio_seg and orig_audio_seg.dBFS != float('-inf')) else -18.0

        extra_refs = []
        if getattr(client, "__class__", None).__name__ == "LocalSpeechClient":
            try:
                # Use 1-2 extra short clips from the same speaker (different segments
                # than the reference) so Fish Speech gets a stronger voice prior.
                speaker_other = [
                    (idx, s) for idx, s in enumerate(segments)
                    if speaker_assignments[idx] == speaker_id and idx != i
                    and s.wav_path and os.path.exists(s.wav_path)
                    and 1500 <= (s.end_ms - s.start_ms) <= 6000
                ]
                for idx, s in speaker_other[:2]:
                    try:
                        extra_refs.append({
                            "audio": _read_wav_bytes(s.wav_path),
                            "text": seg_texts[idx] or "",
                        })
                    except Exception:
                        continue
            except Exception as e:
                logger.warning("Could not build extra references: %s", e)

        # Per-segment emotion takes priority; fall back to job-level emotion.
        effective_emotion = seg_emo if seg_emo and seg_emo != "any" else emotion
        # Adjust the playback ratio window to allow stretching to match slow
        # original speech (low tts_min_playback_ratio) without distorting.
        local_min_ratio = tts_min_playback_ratio
        local_max_ratio = tts_max_playback_ratio

        # Build a tempo-matched TTS prompt: split the translated text into
        # chunks aligned with the original word count, so the TTS rhythm
        # roughly mirrors the source.
        tts_text = uz_text
        if words and len(words) >= 2:
            tts_text = _per_word_tempo_text(uz_text, words, total_target_ms=max_len_ms)

        # Try TTS up to 2 times: if validation fails, fall back to atempo
        # matching by relaxing the chunk_pause and re-synthesizing.
        tts_audio: Optional[AudioSegment] = None
        for attempt in range(2):
            try:
                rendered = _render_tts_segment(
                    client=client,
                    ffmpeg=ffmpeg,
                    tts_dir=tts_dir,
                    segment_key=f"seg_{i:04d}_a{attempt}",
                    text=tts_text,
                    voice=voice,
                    target_ms=max_len_ms,
                    split_max_chars=tts_split_max_chars,
                    chunk_pause_ms=tts_chunk_pause_ms if attempt == 0 else max(60, tts_chunk_pause_ms // 2),
                    min_playback_ratio=local_min_ratio,
                    max_playback_ratio=local_max_ratio,
                    emotion_instructions=emotion_instructions,
                    emotion_speed=emotion_speed,
                    emotion=effective_emotion,
                    reference_audio=ref_audio,
                    reference_text=ref_text,
                    target_dbfs=orig_dbfs,
                    extra_references=extra_refs if extra_refs else None,
                )
                # Validate the rendered audio by sampling a chunk written to disk
                sample_path = os.path.join(tts_dir, f"seg_{i:04d}_a{attempt}_sample.wav")
                try:
                    rendered.export(sample_path, format="wav")
                    with open(sample_path, "rb") as f:
                        sample_bytes = f.read()
                    if _validate_tts_output(sample_bytes):
                        tts_audio = rendered
                        break
                    else:
                        logger.warning("TTS attempt %s failed validation for seg %s", attempt, i)
                except Exception:
                    tts_audio = rendered
                    break
            except Exception as exc:
                logger.error("TTS render failed for seg %s attempt %s: %s", i, attempt, exc)
        if tts_audio is None:
            logger.warning("All TTS attempts failed validation; using silence for seg %s", i)
            tts_audio = AudioSegment.silent(duration=max_len_ms)

        # Inject major-pause timing to roughly match the original cadence.
        if words and len(words) >= 2:
            tts_audio = _inject_tempo_pauses(
                tts_audio, words, seg.start_ms, seg.end_ms
            )

        return i, tts_audio

    if tts_workers <= 1:
        for i, seg in enumerate(segments):
            _, tts_audio = _process_one(i, seg)
            if len(tts_audio) > 0:
                dialog = dialog.overlay(tts_audio, position=seg.start_ms)
    else:
        logger.info("Running TTS for %s segments in parallel (workers=%s)", len(segments), tts_workers)
        results: List[Tuple[int, AudioSegment]] = []
        with ThreadPoolExecutor(max_workers=tts_workers) as ex:
            futures = [ex.submit(_process_one, i, seg) for i, seg in enumerate(segments)]
            for fut in futures:
                try:
                    results.append(fut.result())
                except Exception as exc:
                    logger.error("TTS worker failed: %s", exc)
        # Overlay in original segment order
        for i, tts_audio in sorted(results, key=lambda x: x[0]):
            seg = segments[i]
            if len(tts_audio) > 0:
                dialog = dialog.overlay(tts_audio, position=seg.start_ms)

    # ── Step 5: dialog loudness normalization ──────────────────────────────
    target_lufs = float(os.getenv("DIALOG_TARGET_LUFS", "-16.0"))
    dialog = _loudness_normalize(dialog, target_lufs)

    dialog_path = os.path.join(job_dir, "dialog_uzb.wav")
    dialog.export(dialog_path, format="wav")
    logger.info("Dialog track exported: job_id=%s path=%s", job_id, dialog_path)

    mixed_audio = os.path.join(job_dir, "mixed.m4a")
    logger.info("Mixing background and dubbed dialog: job_id=%s", job_id)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            bg_wav,
            "-i",
            dialog_path,
            "-filter_complex",
            "[1:a]asplit=2[sc][mix];[0:a][sc]sidechaincompress=threshold=0.0625:ratio=4.0:attack=10:release=150[bg];[bg][mix]amix=inputs=2:duration=first:dropout_transition=2",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            mixed_audio,
        ]
    )

    if not has_video:
        logger.info("Dub job finished with audio output: job_id=%s path=%s", job_id, mixed_audio)
        return mixed_audio, "audio"

    output_video = os.path.join(job_dir, "dubbed.mp4")
    logger.info("Muxing final video: job_id=%s", job_id)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-i",
            mixed_audio,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            output_video,
        ]
    )

    logger.info("Dub job finished with video output: job_id=%s path=%s", job_id, output_video)
    return output_video, "video"


def dub_video(
    *,
    client: SpeechClient,
    input_path: str,
    work_dir: str,
    target_lang_hint: str = "uz",
    emotion: Optional[str] = None,
) -> str:
    output_path, _output_kind = dub_media(
        client=client,
        input_path=input_path,
        work_dir=work_dir,
        has_video=True,
        target_lang_hint=target_lang_hint,
        emotion=emotion,
    )
    return output_path
