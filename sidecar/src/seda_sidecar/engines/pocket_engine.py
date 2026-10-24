"""pocket-tts-farsi-v2 engine (CPU) + reference-clip preparation.

Pipeline (mirrors the model's own Space app, training/farsi/v2/space/app.py and
training/farsi/synthesize.py in github.com/mallahyari/pocket-tts, MIT):

  text_fa.normalize_fa_text   digits -> words, letter variants, punctuation, ZWNJ
  pocket_latin                Latin words -> Persian letters (the model's normaliser deletes Latin)
  split into sentences (. ! ؟) and clauses (، ؛ :)
  pocket_normalize_fa         the exact normaliser the model was trained with (vendored)
  Homo-GE2PE G2P              Persian -> romanised phonemes, ezafe marked with "1"
  chunk                       9-18 tokens per chunk, never splitting an ezafe-bound pair,
                              clause ends preferred as breaks
  generate                    per chunk; a chunk whose audio reaches ~the length cap never
                              emitted EOS (runaway) -> retry, then halve and recurse
  trim + join                 trim each chunk's own silence, join with 0.15/0.25/0.30 s gaps
                              (budget split / clause / sentence)

The voice prompt must be <= 5 s (the model was trained with 5 s prompts; longer ones
make it continue the prompt's own sentence).

Weights: mehdi-hf/pocket-tts-farsi-v2 is CC-BY-NC-4.0 (non-commercial);
G2P mehdi-hf/Homo-GE2PE-Persian-HF is MIT (byt5-small base, Apache-2.0).
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .. import config
from ..store import VoiceProfile
from .base import Engine

log = logging.getLogger("seda.pocket")

SR = 24_000
TEMPERATURE = 0.3
EOS_THRESHOLD = -2.0
FRAMES_AFTER_EOS = 0
MAX_TOKENS = 16  # per chunk; training average ~11, clean band 9-16, 21+ stops terminating
MIN_TOKENS = 5
VOICE_SEC = 5.0
JOIN_SEC = 0.15  # split made only to fit the token budget
CLAUSE_SEC = 0.25  # comma / semicolon / colon
PAUSE_SEC = 0.30  # sentence boundary
CAP_RATIO = 0.97  # healthy generations end <= 0.92 of the cap, runaways >= 1.03
RETRIES = 2
MAX_SPLIT_DEPTH = 2
EZAFE_MARK = "1"
# GE2PE's alphabet -> the notation the TTS model was trained on.
TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})
SENTENCE_SPLIT = re.compile(r"(?<=[.!؟])\s+")
CLAUSE_SPLIT = re.compile(r"(?<=[،؛:])\s+")


def _set_threads() -> None:
    import torch

    n = int(os.environ.get("SEDA_POCKET_THREADS", "0")) or min(8, max(1, (os.cpu_count() or 4) // 2))
    try:
        torch.set_num_threads(n)
    except RuntimeError:
        pass


def load_tts_model():
    from pocket_tts import TTSModel

    return TTSModel.load_model(
        config=f"hf://{config.POCKET_MODEL}/model.yaml",
        temp=TEMPERATURE,
        eos_threshold=EOS_THRESHOLD,
    )


# ====================================================================== reference clip
def _read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        import librosa

        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    return a.astype(np.float32)


def _frame_rms(a: np.ndarray, win: int) -> np.ndarray:
    n = len(a) // win
    if n == 0:
        return np.zeros(1, dtype=np.float64)
    f = a[: n * win].astype(np.float64).reshape(n, win)
    return np.sqrt((f**2).mean(axis=1) + 1e-12)


def _best_window(a: np.ndarray, max_sec: float = VOICE_SEC) -> tuple[int, int]:
    """[start, end) sample range: speech onset .. a quiet point at or before max_sec."""
    win = int(0.02 * SR)
    rms = _frame_rms(a, win)
    db = 20 * np.log10(rms)
    floor = np.percentile(db, 10)
    peak = np.percentile(db, 95)
    thr = max(floor + 0.25 * (peak - floor), peak - 40)
    voiced = np.nonzero(db > thr)[0]
    if voiced.size == 0:
        return 0, min(len(a), int(max_sec * SR))
    start = max(0, voiced[0] * win - int(0.08 * SR))
    last = min(len(a), (voiced[-1] + 1) * win + int(0.12 * SR))
    if last - start <= max_sec * SR:
        return start, last
    # cut at the quietest 100 ms stretch between 2.5 s and max_sec after the start
    lo = start + int(2.5 * SR)
    hi = start + int(max_sec * SR)
    f_lo, f_hi = lo // win, min(len(rms), hi // win)
    if f_hi - f_lo < 6:
        return start, hi
    smooth = np.convolve(db, np.ones(5) / 5, mode="same")
    seg = smooth[f_lo:f_hi]
    # prefer later cut points slightly (longer prompt) if almost as quiet
    bias = np.linspace(3.0, 0.0, len(seg))
    k = int(np.argmin(seg + bias))
    cut = (f_lo + k) * win + win // 2
    return start, min(cut, hi)


def score_clip(a: np.ndarray) -> float:
    """Higher is better: duration near 4.5 s, clean (SNR), not clipped, mostly speech."""
    dur = len(a) / SR
    if dur < 1.0:
        return 0.0
    win = int(0.02 * SR)
    db = 20 * np.log10(_frame_rms(a, win))
    snr = float(np.percentile(db, 90) - np.percentile(db, 10))
    s_snr = float(np.clip((snr - 10) / 30, 0, 1))
    s_dur = float(np.clip(1 - abs(dur - 4.5) / 2.5, 0, 1))
    clipped = float(np.mean(np.abs(a) > 0.99))
    s_clip = float(np.clip(1 - clipped * 200, 0, 1))
    thr = np.percentile(db, 10) + 0.3 * (np.percentile(db, 95) - np.percentile(db, 10))
    active = float(np.mean(db > thr))
    s_act = float(np.clip(1 - abs(active - 0.8) / 0.5, 0, 1))
    return round(0.35 * s_dur + 0.35 * s_snr + 0.15 * s_clip + 0.15 * s_act, 4)


def choose_reference(profile: VoiceProfile) -> dict | None:
    """Pick the best <= 5 s window over the profile's samples.

    Returns {sample_id, start, duration, score, audio} or None if there are no samples.
    The profile's explicit reference_sample wins whenever it yields a usable clip.
    """
    best = None
    for s in profile.samples:
        try:
            a = _read_wav(s.wav)
        except Exception as e:  # pragma: no cover
            log.warning("cannot read sample %s: %s", s.id, e)
            continue
        if a.size < SR // 2:
            continue
        st, en = _best_window(a)
        clip = a[st:en]
        sc = score_clip(clip)
        rank = sc + (1.0 if s.id == profile.reference_sample and sc > 0.2 else 0.0)
        if best is None or rank > best["rank"]:
            best = {"sample_id": s.id, "start": round(float(st) / SR, 3), "duration": round(len(clip) / SR, 3),
                    "score": sc, "rank": rank, "audio": clip}
    if best is not None:
        clip = best["audio"]
        peak = float(np.max(np.abs(clip))) or 1.0
        if peak > 0.99:
            clip = clip / peak * 0.99
        fade = int(0.01 * SR)
        if len(clip) > 2 * fade:
            clip = clip.copy()
            clip[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            clip[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        best["audio"] = clip.astype(np.float32)
        best.pop("rank")
    return best


def _write_wav(path: Path, a: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(a, -1, 1), SR, subtype="PCM_16")


def prepare_pocket_reference(profile: VoiceProfile, out_dir: Path, model=None) -> dict:
    """Choose the best clean <= 5 s clip, write out_dir/ref.wav and out_dir/voice.safetensors.

    `model` is an already-loaded pocket TTSModel (optional; otherwise one is loaded on CPU
    just to encode the voice state, then freed). Returns
    {sample_id, start, duration, score, state: bool}. Raises ValueError without samples.
    """
    out_dir = Path(out_dir)
    ref = choose_reference(profile)
    if ref is None:
        raise ValueError("profile has no usable samples for a pocket reference clip")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_wav(out_dir / "ref.wav", ref.pop("audio"))
    ref["state"] = False
    try:
        import torch
        from pocket_tts.models.model_state import export_model_state

        _set_threads()
        own = model is None
        if own:
            eng = _loaded_engine_in_process()
            model = eng._model if eng is not None else load_tts_model()
            own = eng is None
        with torch.no_grad():
            state = model.get_state_for_audio_prompt(out_dir / "ref.wav")
        export_model_state(state, out_dir / "voice.safetensors")
        ref["state"] = True
        if own:
            del model
    except Exception as e:  # the engine re-encodes ref.wav if the state is missing
        log.warning("could not cache pocket voice state: %s", e)
    import json

    (out_dir / "ref.json").write_text(json.dumps(ref, ensure_ascii=False, indent=2), encoding="utf-8")
    return ref


def _loaded_engine_in_process():
    try:
        from .. import manager

        e = manager._engines.get("pocket_v2")
        return e if e is not None and e.loaded else None
    except Exception:
        return None


def _binds(prev: str, nxt: str) -> bool:
    """True if no chunk boundary may fall between these two phoneme words:
    an ezafe ("ketAbe1 man") or the "?o" linking number parts ("bist ?o panj")."""
    return prev.endswith(EZAFE_MARK) or nxt == "?o" or prev == "?o"


def time_stretch(wav: np.ndarray, speed: float) -> np.ndarray:
    """Pitch-preserving tempo change: ffmpeg atempo (WSOLA), librosa as a fallback."""
    import shutil
    import subprocess

    speed = float(np.clip(speed, 0.5, 2.0))
    if shutil.which("ffmpeg"):
        try:
            r = subprocess.run(
                ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                 "-filter:a", f"atempo={speed:.4f}", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"],
                input=wav.astype(np.float32).tobytes(), capture_output=True, timeout=60, check=True,
            )
            out = np.frombuffer(r.stdout, dtype=np.float32)
            if out.size:
                return out.copy()
        except Exception as e:  # pragma: no cover
            log.warning("ffmpeg atempo failed (%s); using librosa", e)
    import librosa

    return librosa.effects.time_stretch(wav, rate=speed).astype(np.float32)


# ====================================================================== silence trim
def trim_silence(a: np.ndarray, keep_ms: float = 120.0) -> np.ndarray:
    """Cut the silence each independently generated chunk opens and closes with.

    From pocket-tts training/farsi/synthesize.py (MIT): a deliberately low threshold
    and a wide margin, so quiet voiced-stop onsets are not eaten.
    """
    if a.size == 0:
        return a
    rms = float(np.sqrt((a.astype(np.float64) ** 2).mean()))
    if rms <= 0:
        return a
    win = int(0.02 * SR)
    fr = _frame_rms(a, win)
    loud = np.nonzero(fr > 0.04 * rms)[0]
    if loud.size == 0:
        return a
    margin = int(keep_ms / 1000.0 * SR)
    return a[max(0, loud[0] * win - margin): min(len(a), (loud[-1] + 1) * win + margin)]


# ====================================================================== engine
class PocketV2Engine(Engine):
    id = "pocket_v2"
    name = "Pocket TTS Farsi v2"
    uses_gpu = False

    def __init__(self) -> None:
        self._model = None
        self._sp = None
        self._g2p_tok = None
        self._g2p = None
        self._states: OrderedDict = OrderedDict()  # key -> voice state
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
            from transformers import AutoTokenizer, T5ForConditionalGeneration

            _set_threads()
            model = load_tts_model()
            tok = AutoTokenizer.from_pretrained(config.G2P_MODEL)
            g2p = T5ForConditionalGeneration.from_pretrained(config.G2P_MODEL, dtype=torch.float32).eval()
            self._sp = model.flow_lm.conditioner.tokenizer.sp
            self._g2p_tok, self._g2p = tok, g2p
            self._model = model

    def unload(self) -> None:
        with self._lock:
            self._model = self._sp = self._g2p = self._g2p_tok = None
            self._states.clear()
            import gc

            gc.collect()

    # ------------------------------------------------------------ text
    def _count(self, s: str) -> int:
        return len(self._sp.encode(s.replace(EZAFE_MARK, "")))

    def phonemise_batch(self, texts: list[str]) -> list[str]:
        """Persian clauses -> phonemes (ezafe marks kept). Empty input -> ""."""
        import torch

        from ..pocket_normalize_fa import normalize_for_model

        prepped: list[str] = []
        for t in texts:
            try:
                t = normalize_for_model(t)
            except ValueError:
                t = ""
            # "؟" and the glottal stop share a symbol in this notation
            prepped.append(t.replace("؟", "").replace("?", "").strip())
        idx = [i for i, t in enumerate(prepped) if t]
        out = [""] * len(texts)
        if not idx:
            return out
        enc = self._g2p_tok([prepped[i] for i in idx], padding=True, add_special_tokens=False,
                            return_attention_mask=True, return_tensors="pt")
        with torch.no_grad():
            gen = self._g2p.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                     num_beams=5, min_length=1, max_length=512, early_stopping=True)
        dec = self._g2p_tok.batch_decode(gen, skip_special_tokens=True)
        for i, d in zip(idx, dec):
            out[i] = d.strip().translate(TO_PHONEMES)
        return out

    def plan(self, text: str) -> list[tuple[str, str]]:
        """Raw text -> [(chunk phonemes with ezafe marks, boundary kind after it)]."""
        from ..pocket_latin import latin_to_persian
        from ..text_fa import normalize_fa_text

        text = latin_to_persian(normalize_fa_text(text))
        sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
        clauses: list[tuple[int, str]] = []
        for si, sent in enumerate(sentences):
            for c in CLAUSE_SPLIT.split(sent):
                if c.strip():
                    clauses.append((si, c))
        phon = self.phonemise_batch([c for _, c in clauses])
        plan: list[tuple[str, str]] = []
        for si in range(len(sentences)):
            words: list[tuple[str, bool]] = []
            for (sj, _), ph in zip(clauses, phon):
                if sj != si:
                    continue
                ws = ph.split()
                words.extend((w, j == len(ws) - 1) for j, w in enumerate(ws))
            cs = self._pack(words)
            if cs:
                plan.extend(cs[:-1])
                plan.append((cs[-1][0], "sentence"))
        return self._merge_tiny(plan)

    def _merge_tiny(self, plan: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Fold chunks under MIN_TOKENS ("salAm") into a neighbour when the result fits.

        A 1-3 token chunk is far below the ~11-token training average; the model then
        tends to mumble or continue the voice prompt. Losing a pause is the lesser evil.
        """
        plan = list(plan)
        i = 0
        while i < len(plan) and len(plan) > 1:
            c, kind = plan[i]
            if self._count(c) >= MIN_TOKENS:
                i += 1
                continue
            cands = []
            if i + 1 < len(plan):
                cands.append((self._count(c) + self._count(plan[i + 1][0]), i))
            if i > 0:
                cands.append((self._count(plan[i - 1][0]) + self._count(c), i - 1))
            cands = [x for x in cands if x[0] <= MAX_TOKENS + 2]
            if not cands:
                i += 1
                continue
            _, j = min(cands)
            plan[j : j + 2] = [(f"{plan[j][0]} {plan[j + 1][0]}", plan[j + 1][1])]
            i = max(0, j)
        return plan

    def _pack(self, words: list[tuple[str, bool]]) -> list[tuple[str, str]]:
        """Chunk a sentence's phoneme words (from the v2 Space's plan_sentence)."""
        if not words:
            return []
        total = self._count(" ".join(w for w, _ in words))
        n_chunks = max(1, math.ceil(total / MAX_TOKENS))
        out: list[tuple[str, str]] = []
        for _ in range(4):
            target = math.ceil(total / n_chunks)
            out, cur = [], ""
            for i, (word, ends_clause) in enumerate(words):
                trial = f"{cur} {word}".strip()
                bound = bool(cur) and _binds(cur.split()[-1], word)
                owed = len(out) < n_chunks - 1
                over = self._count(trial) > MAX_TOKENS
                if cur and not bound and (over or (owed and self._count(trial) > target + 2)):
                    out.append((cur, "budget"))
                    cur = word
                    continue
                cur = trial
                if not ends_clause or i == len(words) - 1:
                    continue
                rest = self._count(" ".join(w for w, _ in words[i + 1:]))
                if self._count(cur) >= MIN_TOKENS and rest >= MIN_TOKENS:
                    out.append((cur, "clause"))
                    cur = ""
            if cur:
                out.append((cur, "clause"))
            if len(out) <= n_chunks:
                break
            n_chunks = len(out)
        # pull words back from the previous chunk rather than leave a starved tail
        for i in range(len(out) - 1, 0, -1):
            while self._count(out[i][0]) < MIN_TOKENS:
                prev = out[i - 1][0].split()
                if len(prev) < 2 or self._count(" ".join(prev[:-1])) < MIN_TOKENS:
                    break
                if _binds(prev[-2], prev[-1]):
                    break
                out[i - 1] = (" ".join(prev[:-1]), out[i - 1][1])
                out[i] = (f"{prev[-1]} {out[i][0]}", out[i][1])
        return out

    # ------------------------------------------------------------ voice
    def _state_for(self, profile: VoiceProfile, run_dir: Path | None):
        pocket = Path(run_dir) / "pocket" if run_dir else None
        if pocket is not None and (pocket / "voice.safetensors").is_file():
            p = pocket / "voice.safetensors"
            key = ("state", str(p), p.stat().st_mtime)
        elif pocket is not None and (pocket / "ref.wav").is_file():
            p = pocket / "ref.wav"
            key = ("wav", str(p), p.stat().st_mtime)
        else:
            p = None
            key = ("profile", profile.id, profile.reference_sample,
                   tuple((s.id, s.wav.stat().st_mtime if s.wav.exists() else 0) for s in profile.samples))
        if key in self._states:
            self._states.move_to_end(key)
            return self._states[key]
        if p is not None:
            state = self._model.get_state_for_audio_prompt(p)
        else:
            ref = choose_reference(profile)
            if ref is None:
                log.warning("profile %s has no samples; using the model's example voice", profile.id)
                ref_audio = self._default_prompt()
            else:
                log.info("pocket voice for %s: sample %s @%.2fs %.2fs score %.3f", profile.id,
                         ref["sample_id"], ref["start"], ref["duration"], ref["score"])
                ref_audio = ref["audio"]
            import torch

            state = self._model.get_state_for_audio_prompt(torch.from_numpy(ref_audio).unsqueeze(0))
        self._states[key] = state
        while len(self._states) > 4:
            self._states.popitem(last=False)
        return state

    def _default_prompt(self) -> np.ndarray:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(config.POCKET_MODEL, "samples/prompt_hello.wav")
        a = _read_wav(Path(p))
        st, en = _best_window(a)
        return a[st:en]

    # ------------------------------------------------------------ generation
    def _cap_seconds(self, text: str) -> float:
        m = self._model
        tokens = m.flow_lm.conditioner.prepare(text).shape[1]
        tps = getattr(m, "_TOKENS_PER_SECOND_ESTIMATE", 3.0)
        pad = getattr(m, "_GEN_SECONDS_PADDING", 2.0)
        fr = m.config.mimi.frame_rate
        return math.ceil((tokens / tps + pad) * fr) / fr

    def _generate_chunk(self, state, text: str, depth: int = 0) -> np.ndarray:
        """One chunk; retry a runaway (no EOS -> audio reaches the cap), then halve it."""
        cap = self._cap_seconds(text)
        shortest = None
        for attempt in range(RETRIES + 1):
            audio = self._model.generate_audio(state, text, frames_after_eos=FRAMES_AFTER_EOS)
            audio = audio.detach().float().cpu().numpy().reshape(-1)
            if len(audio) / SR <= CAP_RATIO * cap:
                return audio
            log.warning("runaway chunk (attempt %d, %.1fs >= cap %.1fs): %s",
                        attempt + 1, len(audio) / SR, cap, text)
            if shortest is None or len(audio) < len(shortest):
                shortest = audio
        words = text.split()
        if depth >= MAX_SPLIT_DEPTH or len(words) < 4:
            return shortest
        half = len(words) // 2
        seam = np.zeros(int(0.05 * SR), dtype=np.float32)
        a = trim_silence(self._generate_chunk(state, " ".join(words[:half]), depth + 1))
        b = trim_silence(self._generate_chunk(state, " ".join(words[half:]), depth + 1))
        return np.concatenate([a, seam, b])

    def synthesize(self, text: str, profile: VoiceProfile, run_dir: Path | None, speed: float = 1.0) -> np.ndarray:
        import torch

        with self._lock, torch.no_grad():
            if self._model is None:
                self.load()
            plan = self.plan(text)
            if not plan:
                return np.zeros(int(0.1 * SR), dtype=np.float32)
            state = self._state_for(profile, run_dir)
            gaps = {k: np.zeros(int(v * SR), dtype=np.float32)
                    for k, v in (("budget", JOIN_SEC), ("clause", CLAUSE_SEC), ("sentence", PAUSE_SEC))}
            pieces: list[np.ndarray] = []
            for i, (chunk, kind) in enumerate(plan):
                spoken = chunk.replace(EZAFE_MARK, "")
                log.debug("chunk %d/%d (%d tok): %s", i + 1, len(plan), self._count(spoken), spoken)
                pieces.append(trim_silence(self._generate_chunk(state, spoken)))
                if i < len(plan) - 1:
                    pieces.append(gaps[kind])
        wav = np.concatenate(pieces).astype(np.float32)
        if speed and abs(speed - 1.0) > 0.01:
            wav = time_stretch(wav, speed)
        # short lead-in / tail so players and ASR do not clip the first/last phoneme
        wav = np.concatenate([np.zeros(int(0.05 * SR), np.float32), wav, np.zeros(int(0.1 * SR), np.float32)])
        return np.clip(wav, -1.0, 1.0)
