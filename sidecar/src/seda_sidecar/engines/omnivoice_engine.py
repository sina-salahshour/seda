"""OmniVoice (k2-fsa/OmniVoice) TTS engine: zero-shot cloning + per-run LoRA adapters.

API used (omnivoice 0.2.x):
  model = OmniVoice.from_pretrained(id, device_map=dev, dtype=fp16, attn_implementation="sdpa")
  prompt = model.create_voice_clone_prompt(ref_audio=wav_path, ref_text=transcript)
  audio = model.generate(text=..., language="fa", voice_clone_prompt=prompt, speed=..)[0]

LoRA adapters (run_dir/adapter, PEFT format written by train_worker) are attached
with ``peft.PeftModel`` *unmerged* and kept side by side as named adapters, so
switching between runs (or back to zero-shot via ``disable_adapter()``) never
reloads the 2.3 GB base model. The LoRA layers are injected into the OmniVoice
modules in place, so ``OmniVoice.generate`` keeps working unchanged.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .. import config
from ..store import SampleRef, VoiceProfile
from .base import Engine

log = logging.getLogger("seda.omnivoice")

LANGUAGE = "fa"
NUM_STEP = int(os.environ.get("SEDA_OMNIVOICE_STEPS", "32"))
GUIDANCE = float(os.environ.get("SEDA_OMNIVOICE_GUIDANCE", "2.0"))
MAX_ADAPTERS = 4
MAX_PROMPTS = 16


def _normalize(text: str) -> str:
    try:
        from ..text_fa import normalize_fa_text
    except ImportError:  # pragma: no cover - B's module missing
        return " ".join(text.replace("ي", "ی").replace("ك", "ک").split())
    return normalize_fa_text(text)


def pick_reference(profile: VoiceProfile) -> SampleRef | None:
    """profile.reference() if set, else the best transcribed 3-10 s clip (prefer 5-8 s)."""
    ref = profile.reference()
    if ref is not None and ref.wav.is_file():
        return ref
    cands = [s for s in profile.samples if s.text.strip() and s.wav.is_file()]
    if not cands:
        return None

    def score(s: SampleRef) -> float:
        d = s.duration
        if 5.0 <= d <= 8.0:
            return 0.0
        if 3.0 <= d <= 10.0:
            return min(abs(d - 5.0), abs(d - 8.0))
        # outside the recommended window: heavily penalised but still usable
        return 10.0 + min(abs(d - 3.0), abs(d - 10.0))

    return min(cands, key=lambda s: (score(s), s.id))


def _clear_higgs_caches() -> None:
    """transformers' Higgs audio tokenizer memoises a method with functools.lru_cache,
    which keeps the tokenizer (~770 MB of CUDA weights) alive after unload. Clear it."""
    try:
        from transformers.models.higgs_audio_v2_tokenizer import modeling_higgs_audio_v2_tokenizer as m
    except Exception:  # pragma: no cover
        return
    for obj in vars(m).values():
        if isinstance(obj, type):
            for attr in vars(obj).values():
                fn = getattr(attr, "cache_clear", None)
                if callable(fn):
                    fn()


class OmniVoiceEngine(Engine):
    id = "omnivoice"
    name = "OmniVoice"
    uses_gpu = True

    def __init__(self) -> None:
        self._model = None  # OmniVoice (LoRA layers injected in place once _peft exists)
        self._peft = None  # peft.PeftModel wrapping self._model, created on first adapter
        self._adapters: OrderedDict[str, str] = OrderedDict()  # adapter name -> cache key
        self._prompts: OrderedDict[tuple, object] = OrderedDict()
        self._device = "cpu"
        self._lock = threading.RLock()

    # ------------------------------------------------------------ lifecycle
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch
            from omnivoice.models.omnivoice import OmniVoice

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if dev == "cuda" else torch.float32
            log.info("loading OmniVoice %s on %s (%s)", config.OMNIVOICE_MODEL, dev, dtype)
            self._model = OmniVoice.from_pretrained(
                config.OMNIVOICE_MODEL,
                device_map=dev,
                dtype=dtype,
                attn_implementation="sdpa",
            )
            self._model.eval()
            self._device = dev

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._peft = None
            self._adapters.clear()
            self._prompts.clear()
            _clear_higgs_caches()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------ adapters
    def _activate(self, adapter_dir: Path | None):
        """Select the LoRA adapter (or none). Returns a context manager for generate()."""
        if adapter_dir is None:
            return self._peft.disable_adapter() if self._peft is not None else nullcontext()
        st = adapter_dir / "adapter_model.safetensors"
        key = f"{adapter_dir.resolve()}:{st.stat().st_mtime_ns if st.exists() else 0}"
        name = "r_" + format(abs(hash(key)) & 0xFFFFFFFF, "08x")
        if name not in self._adapters:
            from peft import PeftModel

            log.info("loading LoRA adapter %s", adapter_dir)
            # The inference model carries the Higgs audio tokenizer as a submodule; its
            # encoder also has q/k/v_proj layers, so hide it while PEFT injects adapters
            # (otherwise it gets zero-initialised LoRA layers that only cost time).
            tok = self._model.audio_tokenizer
            self._model.audio_tokenizer = None
            try:
                if self._peft is None:
                    self._peft = PeftModel.from_pretrained(
                        self._model, str(adapter_dir), adapter_name=name, is_trainable=False
                    )
                else:
                    self._peft.load_adapter(str(adapter_dir), adapter_name=name, is_trainable=False)
            finally:
                self._model.audio_tokenizer = tok
            self._adapters[name] = key
            self._peft.eval()
        self._adapters.move_to_end(name)
        self._peft.set_adapter(name)
        # evict least-recently-used adapters (never the active one)
        while len(self._adapters) > MAX_ADAPTERS:
            old = next(iter(self._adapters))
            if old == name:
                break
            self._adapters.pop(old)
            try:
                self._peft.delete_adapter(old)
            except Exception as e:  # pragma: no cover
                log.warning("could not delete adapter %s: %s", old, e)
        return nullcontext()

    # ------------------------------------------------------------ prompts
    def _prompt(self, ref: SampleRef):
        st = ref.wav.stat()
        key = (str(ref.wav), st.st_mtime_ns, ref.text)
        p = self._prompts.get(key)
        if p is None:
            # The voice prompt is tokenised by the audio tokenizer, which LoRA never touches,
            # so one prompt serves zero-shot and every adapter.
            p = self._model.create_voice_clone_prompt(
                ref_audio=str(ref.wav), ref_text=_normalize(ref.text)
            )
            self._prompts[key] = p
            while len(self._prompts) > MAX_PROMPTS:
                self._prompts.popitem(last=False)
        else:
            self._prompts.move_to_end(key)
        return p

    # ------------------------------------------------------------ synthesis
    def synthesize(
        self,
        text: str,
        profile: VoiceProfile,
        run_dir: Path | None,
        speed: float = 1.0,
    ) -> np.ndarray:
        with self._lock:
            if self._model is None:
                self.load()
            norm = _normalize(text)
            if not norm.strip():
                return np.zeros(int(0.2 * config.SAMPLE_RATE), dtype=np.float32)
            adapter = None
            if run_dir is not None and (Path(run_dir) / "adapter" / "adapter_config.json").is_file():
                adapter = Path(run_dir) / "adapter"
            ref = pick_reference(profile)
            prompt = self._prompt(ref) if ref is not None else None
            kw = {}
            if speed and abs(speed - 1.0) > 1e-3:
                kw["speed"] = float(min(max(speed, 0.5), 2.0))
            ctx = self._activate(adapter)
            with ctx:
                out = self._model.generate(
                    text=norm,
                    language=LANGUAGE,
                    voice_clone_prompt=prompt,
                    num_step=NUM_STEP,
                    guidance_scale=GUIDANCE,
                    **kw,
                )[0]
            audio = np.asarray(out, dtype=np.float32).reshape(-1)
            sr = getattr(self._model, "sampling_rate", config.SAMPLE_RATE) or config.SAMPLE_RATE
            if sr != config.SAMPLE_RATE:  # pragma: no cover - higgs tokenizer is 24 kHz
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
            return np.clip(audio, -1.0, 1.0).astype(np.float32)
