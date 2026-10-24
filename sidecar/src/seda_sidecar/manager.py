"""Model manager: lazy loading, one GPU TTS engine at a time, serialized GPU work.

Usage:
    with manager.tts_engine("omnivoice", busy="tts") as eng:
        audio = eng.synthesize(...)
    with manager.stt(busy="stt") as stt:
        text = stt.transcribe(path)
"""
from __future__ import annotations

import gc
import logging
import threading
from contextlib import contextmanager
from typing import Iterator

log = logging.getLogger("seda.manager")

# Serialises all model loading and inference (GPU and the CPU pocket engine alike).
GPU_LOCK = threading.RLock()
# Set by the training job manager while a training subprocess owns the GPU.
TRAINING = threading.Event()

_busy: str | None = None
_engines: dict = {}
_stt = None


class GpuBusy(Exception):
    """Raised when a GPU model is requested while training owns the GPU."""


def busy() -> str | None:
    if TRAINING.is_set():
        return "training"
    return _busy


def _engine_classes() -> dict:
    from .engines.omnivoice_engine import OmniVoiceEngine
    from .engines.pocket_engine import PocketV2Engine

    return {"omnivoice": OmniVoiceEngine, "pocket_v2": PocketV2Engine}


def engine(eid: str):
    """Return the (possibly unloaded) engine singleton."""
    if eid not in _engines:
        classes = _engine_classes()
        if eid not in classes:
            raise KeyError(f"unknown engine {eid!r}")
        _engines[eid] = classes[eid]()
    return _engines[eid]


def engines_info() -> list[dict]:
    out = []
    for eid in ("omnivoice", "pocket_v2"):
        e = _engines.get(eid)
        name = e.name if e else {"omnivoice": "OmniVoice", "pocket_v2": "Pocket TTS Farsi v2"}[eid]
        out.append({"id": eid, "name": name, "loaded": bool(e and e.loaded)})
    return out


def stt_model():
    global _stt
    if _stt is None:
        from .stt import WhisperSTT

        _stt = WhisperSTT()
    return _stt


def stt_loaded() -> bool:
    return bool(_stt and _stt.loaded)


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


@contextmanager
def tts_engine(eid: str, busy_label: str = "tts") -> Iterator:
    global _busy
    eng = engine(eid)
    if eng.uses_gpu and TRAINING.is_set():
        raise GpuBusy("GPU is busy with a training run; use pocket_v2 or wait")
    with GPU_LOCK:
        _busy = busy_label
        try:
            if eng.uses_gpu:
                for other in _engines.values():
                    if other is not eng and other.uses_gpu and other.loaded:
                        log.info("unloading %s to make room for %s", other.id, eid)
                        other.unload()
                        _free_cuda()
            if not eng.loaded:
                log.info("loading engine %s", eid)
                eng.load()
            yield eng
        finally:
            _busy = None


@contextmanager
def stt(busy_label: str = "stt") -> Iterator:
    """Whisper. While training owns the GPU, the STT model runs on CPU."""
    global _busy
    m = stt_model()
    with GPU_LOCK:
        _busy = busy_label
        try:
            want = "cpu" if TRAINING.is_set() else None
            if not m.loaded or (want and m.device != want):
                if m.loaded:
                    m.unload()
                m.load(device=want)
            yield m
        finally:
            _busy = None


def unload_all() -> None:
    global _busy
    with GPU_LOCK:
        _busy = "unloading"
        try:
            for e in _engines.values():
                if e.loaded:
                    e.unload()
            if _stt is not None and _stt.loaded:
                _stt.unload()
            _free_cuda()
        finally:
            _busy = None
