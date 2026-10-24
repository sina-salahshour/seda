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


def engine(eid: str):
    """Return the (possibly unloaded) engine singleton."""
    if eid not in _engines:
        from .engines import REGISTRY

        if eid not in REGISTRY:
            raise KeyError(f"unknown engine {eid!r}")
        _engines[eid] = REGISTRY[eid].load_class()()
    return _engines[eid]


def engines_info() -> list[dict]:
    from . import models
    from .engines import REGISTRY

    out = []
    for eid, sp in REGISTRY.items():
        e = _engines.get(eid)
        try:
            ready = models.engine_ready(eid)
        except Exception:  # noqa: BLE001
            ready = False
        out.append({
            "id": eid,
            "name": sp.name,
            "loaded": bool(e and e.loaded),
            "uses_gpu": sp.uses_gpu,
            "trainable": sp.trainable,
            "ready": ready,
            "license": sp.license,
            "commercial": sp.commercial,
            "model": models.engine_model_id(eid),
        })
    return out


def stt_model():
    global _stt
    if _stt is None:
        from . import models
        from .stt import WhisperSTT

        _stt = WhisperSTT(model_id=models.stt_path(), generate_kwargs=models.stt_generate_kwargs())
        _stt.catalog_id = models.stt_model_id()
    return _stt


def stt_loaded() -> bool:
    return bool(_stt and _stt.loaded)


def reset_stt() -> None:
    """Drop the STT model (e.g. after stt_model changed); the next use builds a new one."""
    global _stt
    with GPU_LOCK:
        if _stt is not None and _stt.loaded:
            _stt.unload()
        _stt = None
        _free_cuda()


def unload_engine(eid: str) -> None:
    with GPU_LOCK:
        e = _engines.get(eid)
        if e is not None and e.loaded:
            log.info("unloading engine %s", eid)
            e.unload()
            _free_cuda()


def models_in_use() -> set[str]:
    """Model ids (catalog or "local:...") whose weights are loaded right now."""
    out: set[str] = set()
    for e in list(_engines.values()):
        if e.loaded:
            out.update((getattr(e, "loaded_models", None) or {}).keys())
    if _stt is not None and _stt.loaded:
        out.add(getattr(_stt, "catalog_id", "") or "")
    out.discard("")
    return out


def unload_model(mid: str) -> None:
    """Unload whatever uses model `mid` (before deleting it)."""
    for eid, e in list(_engines.items()):
        if e.loaded and mid in (getattr(e, "loaded_models", None) or {}):
            unload_engine(eid)
    if _stt is not None and getattr(_stt, "catalog_id", None) == mid:
        reset_stt()


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
