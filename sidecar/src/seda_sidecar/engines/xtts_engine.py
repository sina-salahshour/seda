"""XTTS engine: ParsVoice-XTTS (XTTS-v2 fine-tuned on Persian) through coqui-tts.

  model = Xtts.init_from_config(cfg); model.load_checkpoint(cfg, checkpoint_dir=dir)
  gpt_cond_latent, speaker_embedding = model.get_conditioning_latents([wav, ...])
  wav = model.inference(text, "fa", gpt_cond_latent, speaker_embedding, speed=..)["wav"]

coqui's tokenizer has no Persian text cleaner (``preprocess_text`` raises for
"fa"), so text is normalised with ``text_fa.normalize_fa_text`` here and the
tokenizer's ``encode`` is wrapped to skip its cleaners for "fa". The language tag
is ``[fa]`` when the checkpoint's vocab has it (otherwise ``[ar]``, logged).

"Training" for this engine = computing speaker conditioning latents from the
profile's best clips (up to ~45 s) once and caching them in
``runs/<rid>/xtts/latents.pt``; without a run, latents are computed on the fly
from the reference sample (+ a few more clips, ~20 s) and kept in a small LRU.
Weights: CPML (Coqui Public Model License), non-commercial use only.
"""
from __future__ import annotations

import gc
import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .. import config
from ..store import SampleRef, VoiceProfile
from .base import Engine

log = logging.getLogger("seda.xtts")

LANGUAGE = "fa"
MAX_CHARS = 180  # per inference call; XTTS gets unstable on long inputs
ZERO_SHOT_SECONDS = 20.0
ZERO_SHOT_CLIPS = 4
MAX_LATENTS = 8
JOIN_SEC = 0.2
LATENTS_FILE = Path("xtts") / "latents.pt"
_SPLIT = re.compile(r"(?<=[.!?؟؛;\n])\s+")
_CLAUSE = re.compile(r"(?<=[،,:])\s+")


def _normalize(text: str) -> str:
    from ..text_fa import normalize_fa_text

    return normalize_fa_text(text)


def split_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Sentences, then clauses, then words, packed into pieces of <= max_chars."""
    out: list[str] = []
    for sent in (s.strip() for s in _SPLIT.split(text)):
        if not sent:
            continue
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        cur = ""
        for part in (p.strip() for p in _CLAUSE.split(sent)):
            words = part.split() if len(part) > max_chars else [part]
            for w in words:
                if cur and len(cur) + 1 + len(w) > max_chars:
                    out.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}" if cur else w
        if cur:
            out.append(cur)
    return out


def _read(path: Path) -> np.ndarray:
    import soundfile as sf

    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        import librosa

        a = librosa.resample(a, orig_sr=sr, target_sr=config.SAMPLE_RATE)
    return a


def pick_clips(profile: VoiceProfile, max_seconds: float, max_clips: int) -> list[SampleRef]:
    """The profile's reference sample first, then the cleanest 3-15 s clips, up to max_seconds."""
    from .pocket_engine import _best_window, score_clip

    scored: list[tuple[float, SampleRef]] = []
    for s in profile.samples:
        if not s.wav.is_file() or s.duration < 1.0:
            continue
        try:
            a = _read(s.wav)
        except Exception as e:  # noqa: BLE001
            log.warning("cannot read sample %s: %s", s.id, e)
            continue
        st, en = _best_window(a)
        q = score_clip(a[st:en])
        d = s.duration
        q += 0.2 if 3.0 <= d <= 15.0 else 0.0
        q += 2.0 if s.id == profile.reference_sample else 0.0
        scored.append((q, s))
    scored.sort(key=lambda t: (-t[0], t[1].id))
    out: list[SampleRef] = []
    total = 0.0
    for _, s in scored:
        if len(out) >= max_clips or total >= max_seconds:
            break
        out.append(s)
        total += min(s.duration, 30.0)
    return out


def _model_dir(path: str) -> Path:
    """A local checkpoint dir for `path` (dir, or repo id downloaded with the stored token)."""
    p = Path(path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    from .. import catalog, models

    m = next((c for c in catalog.CATALOG.values() if c.repo == path), None)
    try:
        return Path(snapshot_download(
            repo_id=path,
            allow_patterns=list(m.allow_patterns) if m and m.allow_patterns else None,
            token=models.hf_token(),
        ))
    except Exception as e:  # noqa: BLE001
        names = {c.__name__ for c in type(e).__mro__}
        status = getattr(getattr(e, "response", None), "status_code", None)
        if "GatedRepoError" in names or status in (401, 403):
            raise models.ModelError(models.GATED_MSG) from e
        raise


def build_model(path: str, device: str):
    """Load an XTTS checkpoint dir (config.json + model.pth + vocab.json) on `device`."""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    d = _model_dir(path)
    cfg = XttsConfig()
    cfg.load_json(str(d / "config.json"))
    model = Xtts(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=str(d), eval=True, use_deepspeed=False)
    model.to(device)
    model.eval()
    lang_tag, keep_zwnj = _patch_tokenizer(model)
    return model, lang_tag, keep_zwnj


def _patch_tokenizer(model) -> tuple[str, bool]:
    tok = model.tokenizer
    vocab = tok.tokenizer.get_vocab()
    if f"[{LANGUAGE}]" in vocab:
        tag = LANGUAGE
    else:
        tag = "ar"
        log.warning("XTTS vocab has no [fa] token; tagging Persian text as [ar]")
    orig_encode = tok.encode

    def encode(txt, lang):
        if lang.split("-")[0] != LANGUAGE:
            return orig_encode(txt, lang)
        tok.check_input_length(txt, lang)
        return tok.tokenizer.encode(f"[{tag}]{txt}".replace(" ", "[SPACE]")).ids

    tok.encode = encode
    tok.char_limits.setdefault(LANGUAGE, MAX_CHARS + 20)
    keep_zwnj = tok.tokenizer.token_to_id("‌") is not None
    return tag, keep_zwnj


def conditioning(model, clips: list[Path], chunk_len: int = 6):
    """(gpt_cond_latent, speaker_embedding) over all clips (each truncated to 30 s)."""
    import math

    total = 0.0
    for c in clips:
        try:
            import soundfile as sf

            total += min(sf.info(str(c)).duration, 30.0)
        except Exception:  # noqa: BLE001
            total += 10.0
    cond_len = max(int(chunk_len), int(math.ceil(total)))
    return model.get_conditioning_latents(
        audio_path=[str(c) for c in clips],
        max_ref_length=30,
        gpt_cond_len=cond_len,
        gpt_cond_chunk_len=int(chunk_len),
    )


class XttsEngine(Engine):
    id = "xtts"
    name = "ParsVoice XTTS"
    uses_gpu = True

    def __init__(self) -> None:
        self._model = None
        self._tag = LANGUAGE
        self._keep_zwnj = True
        self._device = "cpu"
        self._latents: OrderedDict[tuple, tuple] = OrderedDict()
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

            from .. import models

            ids = models.engine_model_ids(self.id)
            path = models.resolve(ids["tts"])
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("loading XTTS %s on %s", path, dev)
            self._model, self._tag, self._keep_zwnj = build_model(path, dev)
            self._device = dev
            self.loaded_models = {ids["tts"]: path}

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self.loaded_models = {}
            self._latents.clear()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------ voice
    def _voice(self, profile: VoiceProfile, run_dir: Path | None):
        import torch

        f = Path(run_dir) / LATENTS_FILE if run_dir is not None else None
        if f is not None and f.is_file():
            key = ("run", str(f), f.stat().st_mtime_ns)
        else:
            f = None
            key = ("profile", profile.id, profile.reference_sample,
                   tuple((s.id, s.wav.stat().st_mtime_ns if s.wav.exists() else 0) for s in profile.samples))
        hit = self._latents.get(key)
        if hit is not None:
            self._latents.move_to_end(key)
            return hit
        if f is not None:
            d = torch.load(str(f), map_location="cpu", weights_only=True)
            v = (d["gpt_cond_latent"], d["speaker_embedding"])
        else:
            clips = pick_clips(profile, ZERO_SHOT_SECONDS, ZERO_SHOT_CLIPS)
            if not clips:
                raise ValueError("the profile has no usable samples")
            log.info("xtts zero-shot voice for %s from %s", profile.id, [c.id for c in clips])
            g, s = conditioning(self._model, [c.wav for c in clips])
            v = (g.cpu(), s.cpu())
        self._latents[key] = v
        while len(self._latents) > MAX_LATENTS:
            self._latents.popitem(last=False)
        return v

    # ------------------------------------------------------------ synthesis
    def synthesize(self, text: str, profile: VoiceProfile, run_dir: Path | None, speed: float = 1.0) -> np.ndarray:
        import torch

        with self._lock:
            if self._model is None:
                self.load()
            norm = _normalize(text)
            if not self._keep_zwnj:
                norm = norm.replace("‌", "")
            pieces = split_text(norm)
            sr_out = config.SAMPLE_RATE
            if not pieces:
                return np.zeros(int(0.2 * sr_out), dtype=np.float32)
            gpt, spk = self._voice(profile, run_dir)
            m = self._model
            cfg = m.config
            sr = int(getattr(getattr(cfg, "audio", None), "output_sample_rate", 24000) or 24000)
            gap = np.zeros(int(JOIN_SEC * sr_out), dtype=np.float32)
            out: list[np.ndarray] = []
            for i, piece in enumerate(pieces):
                with torch.inference_mode():
                    res = m.inference(
                        piece, LANGUAGE, gpt, spk,
                        temperature=float(getattr(cfg, "temperature", 0.75)),
                        length_penalty=float(getattr(cfg, "length_penalty", 1.0)),
                        repetition_penalty=float(getattr(cfg, "repetition_penalty", 10.0)),
                        top_k=int(getattr(cfg, "top_k", 50)),
                        top_p=float(getattr(cfg, "top_p", 0.85)),
                        speed=float(min(max(speed or 1.0, 0.5), 2.0)),
                        enable_text_splitting=False,
                    )
                wav = np.asarray(res["wav"], dtype=np.float32).reshape(-1)
                if sr != sr_out:
                    import librosa

                    wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_out)
                if i:
                    out.append(gap)
                out.append(wav)
            return np.clip(np.concatenate(out), -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------ training
    def artifact_ok(self, run_dir: Path) -> bool:
        return (Path(run_dir) / LATENTS_FILE).is_file()

    def train(self, job: dict, run_dir: Path, emit) -> None:
        """Compute and cache conditioning latents from the profile's best clips."""
        import torch

        from .. import models, store

        p = job.get("params") or {}
        path = (job.get("model_paths") or {}).get("tts") or models.resolve(models.engine_model_id(self.id))
        dev = os.environ.get("SEDA_XTTS_TRAIN_DEVICE", "cpu")
        emit(stage="model", progress=0.05, message="loading XTTS")
        print(f"loading XTTS {path} on {dev}", flush=True)
        model, _, _ = build_model(path, dev)
        emit(stage="latents", progress=0.6, message="computing voice latents")
        profile = store.voice_profile(job["profile_id"])
        clips = pick_clips(profile, float(p.get("max_reference_seconds", 45.0)), int(p.get("max_clips", 12)))
        if not clips:
            raise ValueError("profile has no usable samples for XTTS voice latents")
        g, s = conditioning(model, [c.wav for c in clips], int(p.get("gpt_cond_chunk_len", 6)))
        out = Path(run_dir) / LATENTS_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        torch.save({
            "gpt_cond_latent": g.detach().cpu(),
            "speaker_embedding": s.detach().cpu(),
            "samples": [c.id for c in clips],
            "seconds": float(round(sum(min(c.duration, 30.0) for c in clips), 2)),
        }, str(tmp))
        tmp.replace(out)
        print(f"xtts latents from {len(clips)} clips: {[c.id for c in clips]}", flush=True)
        emit(stage="save", progress=0.97, message="saved voice latents")
