"""Audio import and conversion helpers.

Everything here works on float32 mono numpy arrays in [-1, 1]. Decoding goes
through ffmpeg so any format it reads (wav, mp3, m4a, ogg, opus, flac, video
containers...) can be imported.
"""
from __future__ import annotations

import io
import logging
import shutil
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config

log = logging.getLogger("seda.audio")

SR = config.SAMPLE_RATE  # 24 kHz
STT_SR = 16_000

# Import pipeline parameters.
SPLIT_IF_LONGER_S = 30.0  # files longer than this are split ...
MAX_CHUNK_S = 20.0  # ... into chunks of at most this length
MIN_CHUNK_S = 6.0  # (cuts are searched for in [MIN_CHUNK_S, MAX_CHUNK_S])
MIN_KEEP_S = 0.8  # chunks shorter than this after trimming are dropped
TARGET_RMS_DBFS = -20.0  # speech loudness target
PEAK_CEILING = 10 ** (-1.0 / 20)  # -1 dBFS

_FRAME_S = 0.02


class AudioError(ValueError):
    """Unreadable or empty audio (maps to HTTP 400)."""


def ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or "/usr/bin/ffmpeg"


# ---------------------------------------------------------------- decode / encode
def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Decode any ffmpeg-readable file to float32 mono at `sr`."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"file not found: {path}")
    cmd = [
        ffmpeg_bin(), "-nostdin", "-v", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as e:  # pragma: no cover
        raise AudioError("ffmpeg not found") from e
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioError(f"cannot decode {path.name}: {msg[-1] if msg else 'ffmpeg failed'}")
    x = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if x.size == 0:
        raise AudioError(f"no audio in {path.name}")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).round().astype(np.int16)


def write_wav(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    """Write 16-bit PCM mono wav (atomically)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    sf.write(str(tmp), to_int16(x), sr, subtype="PCM_16", format="WAV")
    tmp.replace(path)


def wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    """float32 mono -> 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, to_int16(x), sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return np.asarray(x, dtype=np.float32)
    from scipy.signal import resample_poly

    g = gcd(sr_from, sr_to)
    return resample_poly(np.asarray(x, dtype=np.float32), sr_to // g, sr_from // g).astype(np.float32)


def to_16k(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Resample for Whisper."""
    return resample(x, sr, STT_SR)


def load_16k(path: str | Path) -> np.ndarray:
    return load_audio(path, STT_SR)


# ---------------------------------------------------------------- analysis
def _frame_db(x: np.ndarray, sr: int, frame_s: float = _FRAME_S) -> np.ndarray:
    n = max(1, int(sr * frame_s))
    m = len(x) // n
    if m == 0:
        return np.array([20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9)])
    fr = x[: m * n].reshape(m, n)
    return 20 * np.log10(np.sqrt(np.mean(fr**2, axis=1)) + 1e-9)


def _silence_threshold(db: np.ndarray) -> float:
    """Frames below this level count as silence: 35 dB under the loud speech level,
    but never above -35 dBFS nor below -60 dBFS (noise floor)."""
    loud = float(np.percentile(db, 95))
    return float(np.clip(loud - 35.0, -60.0, -35.0))


def trim_silence(x: np.ndarray, sr: int = SR, pad_s: float = 0.15) -> np.ndarray:
    """Remove leading/trailing silence, keeping `pad_s` of margin."""
    if x.size == 0:
        return x
    db = _frame_db(x, sr)
    voiced = np.flatnonzero(db > _silence_threshold(db))
    if voiced.size == 0:
        return x[:0]
    n = int(sr * _FRAME_S)
    pad = int(sr * pad_s)
    start = max(0, voiced[0] * n - pad)
    end = min(len(x), (voiced[-1] + 1) * n + pad)
    return x[start:end]


def normalize_loudness(x: np.ndarray, sr: int = SR, target_dbfs: float = TARGET_RMS_DBFS) -> np.ndarray:
    """Scale so the RMS of the speech frames is `target_dbfs`, without letting the
    peak exceed -1 dBFS (no clipping, no limiter)."""
    if x.size == 0:
        return x
    x = x - float(np.mean(x))  # remove DC
    db = _frame_db(x, sr)
    active = db[db > _silence_threshold(db)]
    if active.size == 0:
        return x.astype(np.float32)
    # mean power of active frames
    rms_db = 10 * np.log10(np.mean(10 ** (active / 10)) + 1e-12)
    gain = 10 ** ((target_dbfs - rms_db) / 20)
    peak = float(np.max(np.abs(x))) + 1e-9
    gain = min(gain, PEAK_CEILING / peak)
    return (x * gain).astype(np.float32)


def split_on_silence(
    x: np.ndarray,
    sr: int = SR,
    max_s: float = MAX_CHUNK_S,
    min_s: float = MIN_CHUNK_S,
) -> list[np.ndarray]:
    """Cut `x` into chunks of at most `max_s`, at the quietest point (smoothed over
    ~300 ms) between `min_s` and `max_s` after the previous cut."""
    n = int(sr * _FRAME_S)
    db = _frame_db(x, sr)
    # smoothed energy: prefer cutting in the middle of a pause, not a short dip
    win = max(1, int(0.3 / _FRAME_S))
    smooth = np.convolve(10 ** (db / 10), np.ones(win) / win, mode="same")
    total = len(db)
    max_f, min_f = int(max_s / _FRAME_S), int(min_s / _FRAME_S)
    chunks, start = [], 0
    while total - start > max_f:
        lo, hi = start + min_f, min(start + max_f, total)
        seg = smooth[lo:hi]
        # among frames within 1 dB of the quietest, take the latest (longer chunks)
        floor = seg.min()
        cand = np.flatnonzero(seg <= floor * 1.26)
        cut = lo + int(cand[-1])
        chunks.append(x[start * n : cut * n])
        start = cut
    chunks.append(x[start * n :])
    return chunks


# ---------------------------------------------------------------- pipeline
def prepare_import(path: str | Path) -> list[np.ndarray]:
    """Decode -> 24 kHz mono -> trim -> (split if > 30 s) -> trim each -> normalise.

    Returns a list of float32 chunks (at least one), each >= MIN_KEEP_S.
    """
    x = load_audio(path, SR)
    x = trim_silence(x, SR)
    if len(x) / SR < 0.3:
        raise AudioError("the file contains only silence")
    parts = split_on_silence(x, SR) if len(x) / SR > SPLIT_IF_LONGER_S else [x]
    out = []
    for p in parts:
        p = trim_silence(p, SR)
        if len(p) / SR < MIN_KEEP_S and len(parts) > 1:
            continue
        out.append(normalize_loudness(p, SR))
    if not out:
        raise AudioError("no usable speech found")
    log.info("imported %s: %d chunk(s), %s s", Path(path).name, len(out), [round(len(c) / SR, 2) for c in out])
    return out


def duration(x: np.ndarray, sr: int = SR) -> float:
    return round(len(x) / sr, 3)
