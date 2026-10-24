"""Whisper speech-to-text (Persian), lazily loaded.

Long audio (> 30 s) is cut on silences into <= 28 s pieces which are decoded as
one batch, so there is no 30 s truncation.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from . import audio, config

log = logging.getLogger("seda.stt")

_WINDOW_S = 30.0
_PIECE_S = 28.0
_BATCH = 4


class WhisperSTT:
    def __init__(self, model_id: str = config.WHISPER_MODEL, language: str = "fa"):
        self.model_id = model_id
        self.language = language
        self.model = None
        self.processor = None
        self.device: str | None = None
        self.dtype = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self, device: str | None = None) -> None:
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.loaded:
            if self.device == device:
                return
            self.unload()
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        t0 = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
        )
        model.to(device).eval()
        self.model, self.device, self.dtype = model, device, dtype
        if device.startswith("cuda"):
            # The first CUDA generate() pays ~15 s of one-time kernel setup; do it now.
            self._decode([np.zeros(audio.STT_SR, dtype=np.float32)])
        log.info("whisper %s loaded on %s (%s) in %.1fs", self.model_id, device, dtype, time.monotonic() - t0)

    def unload(self) -> None:
        if self.model is None:
            return
        on_cuda = bool(self.device and self.device.startswith("cuda"))
        self.model = None
        self.processor = None
        self.device = None
        import gc

        gc.collect()
        if on_cuda:
            import torch

            torch.cuda.empty_cache()
        log.info("whisper unloaded")

    # ------------------------------------------------------------------
    def transcribe(self, src: str | Path | np.ndarray) -> str:
        """Transcribe a file path (any format ffmpeg reads) or a float32 16 kHz array."""
        if not self.loaded:
            self.load()
        if isinstance(src, (str, Path)):
            x = audio.load_16k(src)
        else:
            x = np.asarray(src, dtype=np.float32).reshape(-1)
        if x.size < audio.STT_SR * 0.1:
            return ""
        if len(x) / audio.STT_SR <= _WINDOW_S:
            pieces = [x]
        else:
            pieces = audio.split_on_silence(x, audio.STT_SR, max_s=_PIECE_S, min_s=10.0)
        texts: list[str] = []
        for i in range(0, len(pieces), _BATCH):
            texts.extend(self._decode(pieces[i : i + _BATCH]))
        return " ".join(t for t in (s.strip() for s in texts) if t)

    def _decode(self, batch: list[np.ndarray]) -> list[str]:
        import torch

        feats = self.processor(
            batch, sampling_rate=audio.STT_SR, return_tensors="pt", return_attention_mask=True
        )
        inputs = feats.input_features.to(self.device, dtype=self.dtype)
        mask = feats.get("attention_mask")
        kwargs = {"attention_mask": mask.to(self.device)} if mask is not None else {}
        with torch.inference_mode():
            ids = self.model.generate(
                inputs,
                language=self.language,
                task="transcribe",
                **kwargs,
            )
        return self.processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
