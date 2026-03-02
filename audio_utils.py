#!/usr/bin/env python3
"""
Audio capture from RTSP and loud sound detection (cry, scream, alarm).
Requires ffmpeg installed. Sends short clip when loud sound detected.
"""

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CHANNELS = 1
# RMS threshold: cry/scream typically > 0.02-0.05 in normalized int16
LOUD_THRESHOLD = 0.03
# Sub-chunk size for detection (0.5 sec)
CHUNK_SAMPLES = SAMPLE_RATE // 2
# Min duration of loud to trigger (consecutive chunks)
MIN_LOUD_CHUNKS = 2


def capture_audio_chunk(rtsp_url: str, duration_sec: float = 5.0) -> bytes | None:
    """
    Capture audio from RTSP stream using ffmpeg. Returns raw PCM (s16le) or None.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-t", str(duration_sec),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-f", "s16le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=int(duration_sec) + 10,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        return proc.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def compute_rms(audio_bytes: bytes) -> float:
    """Compute RMS of raw PCM s16le. Returns normalized value 0-1."""
    if len(audio_bytes) < 2:
        return 0.0
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(arr) == 0:
        return 0.0
    rms = np.sqrt(np.mean(arr.astype(np.float64) ** 2))
    return rms / 32768.0  # Normalize by max int16


def detect_loud_sound(audio_bytes: bytes) -> tuple[bool, float]:
    """
    Detect if audio contains loud sound (cry, scream, alarm) via RMS.
    Returns (has_loud, max_rms).
    """
    if len(audio_bytes) < CHUNK_SAMPLES * 2:  # Need at least 1 chunk
        return False, 0.0
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    n_chunks = len(arr) // CHUNK_SAMPLES
    if n_chunks == 0:
        return False, 0.0
    max_rms = 0.0
    loud_count = 0
    for i in range(n_chunks):
        chunk = arr[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES]
        rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) / 32768.0
        max_rms = max(max_rms, rms)
        if rms >= LOUD_THRESHOLD:
            loud_count += 1
        else:
            loud_count = 0
        if loud_count >= MIN_LOUD_CHUNKS:
            return True, max_rms
    return max_rms >= LOUD_THRESHOLD, max_rms


def save_clip_to_ogg(audio_bytes: bytes) -> bytes | None:
    """Convert raw PCM s16le to OGG for Telegram."""
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
        f.write(audio_bytes)
        raw_path = f.name
    ogg_path = raw_path.replace(".raw", ".ogg")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "-i", raw_path,
            "-c:a", "libopus",
            "-b:a", "32k",
            "-f", "ogg",
            ogg_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=10, check=True)
        with open(ogg_path, "rb") as f:
            data = f.read()
        return data
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    finally:
        Path(raw_path).unlink(missing_ok=True)
        Path(ogg_path).unlink(missing_ok=True)


def check_rtsp_has_audio(rtsp_url: str) -> bool:
    """Quick check if RTSP stream has audio track."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-rtsp_transport", "tcp",
                "-select_streams", "a", "-show_entries", "stream=codec_type",
                "-of", "csv=p=0", "-i", rtsp_url,
            ],
            capture_output=True,
            timeout=15,
        )
        return proc.returncode == 0 and proc.stdout and b"audio" in proc.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
