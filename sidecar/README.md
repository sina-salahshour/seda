# seda-sidecar

Python ML server for Seda (صدا): Persian TTS with voice cloning, Whisper STT and
per-profile LoRA training. The Rust app talks to it over HTTP; the contract is
`../SPEC.md`.

| Role | Model | Device |
|---|---|---|
| Main TTS + LoRA training | `k2-fsa/OmniVoice` (+ `eustlb/higgs-audio-v2-tokenizer`) | GPU, fp16 inference / bf16 training |
| Fast TTS | `mehdi-hf/pocket-tts-farsi-v2` + G2P `mehdi-hf/Homo-GE2PE-Persian-HF` | CPU |
| STT | `openai/whisper-large-v3-turbo`, language `fa` | GPU fp16 (CPU while training) |

## Setup

Requires uv, ffmpeg and an NVIDIA driver recent enough for CUDA 12.8 (the
RTX 50xx / sm_120 needs torch built for cu128; `pyproject.toml` pins
`torch==2.8.0` from the cu128 index).

```sh
cd sidecar
uv sync                      # creates .venv with Python 3.12
# optional: pre-download the models (~6 GB) into the HF cache
for m in k2-fsa/OmniVoice eustlb/higgs-audio-v2-tokenizer mehdi-hf/pocket-tts-farsi-v2 \
         mehdi-hf/Homo-GE2PE-Persian-HF openai/whisper-large-v3-turbo; do
  uv run hf download $m; done
```

No gated models are needed (`kyutai/pocket-tts` is not used).

## Run

```sh
uv run seda-sidecar          # 127.0.0.1:8765
```

Environment:

| Var | Default | |
|---|---|---|
| `SEDA_HOST` / `SEDA_PORT` | `127.0.0.1` / `8765` | bind address |
| `SEDA_DATA` | `~/.local/share/seda` | data dir (sidecar owns `profiles/`) |
| `SEDA_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama, asked to unload before training |
| `SEDA_OMNIVOICE_STEPS` / `SEDA_OMNIVOICE_GUIDANCE` | `32` / `2.0` | OmniVoice diffusion steps / CFG |
| `SEDA_POCKET_THREADS` | `min(8, cores/2)` | CPU threads for pocket_v2 |
| `SEDA_*_MODEL` | see `config.py` | override model ids / local paths |

Quick check:

```sh
curl -s localhost:8765/health
P=$(curl -s -XPOST localhost:8765/profiles -H 'content-type: application/json' -d '{"name":"من"}' | jq -r .id)
curl -s -XPOST localhost:8765/profiles/$P/samples -H 'content-type: application/json' -d '{"wav_path":"/abs/clip.wav"}'
curl -s -XPOST localhost:8765/tts -H 'content-type: application/json' \
  -d "{\"profile_id\":\"$P\",\"text\":\"سلام! ساعت ۸:۳۰ است.\"}" -o out.wav
```

## Code layout (`src/seda_sidecar/`)

| File | |
|---|---|
| `server.py` | FastAPI app, every endpoint in SPEC.md, `main()` |
| `store.py` | profiles / samples / runs JSON on disk (atomic writes) |
| `manager.py` | lazy model loading, one GPU TTS engine at a time, global lock serialising inference, `unload_all()` |
| `engines/base.py` | `Engine`: `load / unload / synthesize(text, profile, run_dir, speed) -> float32 @ 24 kHz` |
| `engines/omnivoice_engine.py` | OmniVoice zero-shot cloning + PEFT LoRA adapters |
| `engines/pocket_engine.py` | pocket-tts-farsi-v2 pipeline + `prepare_pocket_reference` |
| `text_fa.py` | Persian TTS normalisation (digits → words, ZWNJ, ي/ك → ی/ک, punctuation) |
| `pocket_normalize_fa.py`, `pocket_latin.py` | model-specific normaliser (vendored from the fork, MIT) and Latin → Persian respelling |
| `audio.py` | ffmpeg import → 24 kHz mono, silence trim, loudness normalisation (-20 dBFS RMS, peak ≤ -1 dBFS), split > 30 s |
| `stt.py` | Whisper large-v3-turbo (`fa`), long-form by chunking on silences |
| `training.py` / `train_worker.py` | job queue + training subprocess |
| `prompts.py`, `data/prompts_fa.txt` | 201 read-aloud sentences for `GET /prompts` |

## API notes (behaviour the SPEC leaves open)

- Errors are always `{"detail": "<string>"}`: 400 validation (also for what
  FastAPI would report as 422), 404 unknown id, 409 busy (training queue full,
  or OmniVoice requested while training owns the GPU), 503 CUDA out of memory
  (models are unloaded; retry), 500 otherwise.
- `POST /samples`: `source` defaults to `imported`. A file > 30 s is split on
  silences into ≤ 20 s chunks; every chunk is transcribed with Whisper (a given
  `text` is ignored for split files, since it can't be divided reliably).
- `PATCH /profiles`: `active_run` must be a run with status `done`;
  `reference_sample` must exist.
- `POST /tts`: `speed` ∈ [0.5, 2.0]. A run is only used when its status is
  `done`; engines pick their own artefacts from the run dir (`adapter/` for
  OmniVoice, `pocket/` for pocket_v2) and fall back to zero-shot otherwise.
  A profile without samples → 400.
- Training: one job runs, at most one more waits (a third → 409). The worker is
  a separate process (`python -m seda_sidecar.train_worker <run_dir>`); its
  `@@PROGRESS {json}` lines are mirrored into `run.json` and everything goes to
  `train.log`. `cancel` kills the process group and returns the run once it is
  `cancelled`. On sidecar restart, stale queued/running runs become `failed`.
- While a training run owns the GPU, OmniVoice `/tts` → 409, pocket_v2 keeps
  working (CPU) and `/stt` moves Whisper to the CPU (much slower).

## Engines

**OmniVoice.** `OmniVoice.from_pretrained(..., dtype=fp16, attn_implementation="sdpa")`,
`generate(text, language="fa", voice_clone_prompt=..., num_step=32,
guidance_scale=2.0, speed=...)`. The reference is the profile's
`reference_sample`, otherwise the best transcribed 5–8 s sample (then 3–10 s).
LoRA adapters are loaded unmerged with PEFT as named adapters (LRU of 4), so
switching runs or going back to zero-shot (`disable_adapter()`) never reloads
the 2.3 GB base model.

**pocket_v2.** `normalize_fa_text` → Latin respelling → the model's own
`normalize_fa` → Homo-GE2PE G2P (batched, 5 beams) → chunks of 9–16 tokens
(≤ 16, never splitting ezafe pairs, tiny chunks merged) → generation
(temperature 0.3, eos threshold −2) → runaway chunks (hit the length cap) are
retried, then halved → silence-trimmed chunks joined with 0.25 s (clauses),
0.15 s (budget splits) or 0.3 s (sentences) pauses → optional `atempo` speed.
The voice prompt is a ≤ 5 s clip: "training" for this engine
(`prepare_pocket_reference`) scores every sample's best ≤ 5 s window (SNR,
clipping, speech activity, length near 4.5 s) and caches `ref.wav` +
`voice.safetensors` in `runs/<rid>/pocket/`. Noisy or music-backed clips make
poor references; record in a quiet room.

## Training (OmniVoice LoRA)

The installed `omnivoice` 0.2.1 wheel has no LoRA option, so the worker builds
the model with OmniVoice's own builder and wraps it with PEFT using the
settings from upstream `docs/lora_finetuning.md`: r 16, alpha 32, dropout 0.05
on q/k/v/o/gate/up/down, and `audio_embeddings` + `audio_heads` trained fully
(26.9 M trainable params, adapter ≈ 103 MB). Data is prepared the way
`docs/data_preparation.md` describes: text normalised with `normalize_fa_text`
(`language_id` `fa`), audio tokenised with the Higgs audio tokenizer into
`runs/<rid>/data/`.

Defaults (fit in 8 GB): bf16 autocast, sdpa, gradient checkpointing,
`batch_tokens` 3072 × grad-accum 2, AdamW lr 1e-4, cosine with 5 % warmup,
grad-clip 1.0. Default steps = `clamp(150 + 12 × minutes of audio, 200, 1000)`
(5 min → 210, 30 min → 510, 60 min → 870, i.e. roughly 40–60 epochs).
`from_run` continues from that run's adapter.

Before a job starts the sidecar unloads its own models and asks Ollama to
unload every running model (`GET /api/ps`, then `keep_alive: 0`).

## VRAM and speed (RTX 5070 Laptop, 8 GB)

| | VRAM (nvidia-smi) | Speed |
|---|---|---|
| OmniVoice inference | ~3.8 GB (3.5 GB peak allocated) | RTF ≈ 0.22–0.27 zero-shot, ≈ 0.32 with an unmerged LoRA; first call after load warms up for ~15 s |
| Whisper large-v3-turbo fp16 | ~2.0 GB (1.7 GB peak allocated) | ~0.04× real time |
| OmniVoice + Whisper together | ~5.5–6 GB | |
| pocket_v2 | CPU only (~1.5 GB RAM) | RTF ≈ 0.35–0.45 on 8 threads |
| LoRA training | ~3.8 GB (3.0–3.5 GB peak allocated) | 0.7–1.0 s/step, + ~20 s start-up |

Only one sidecar should use the GPU at a time: two instances (or Ollama
holding VRAM) can push the total past 8 GB, in which case requests fail with
503 and training runs fail with "out of GPU memory".

## Licensing

- This sidecar's code: same license as the Seda repo.
- `k2-fsa/OmniVoice` code: Apache-2.0. The HF model card of the weights
  carries no license field (SPEC.md lists Apache-2.0); check before
  redistributing.
- **`mehdi-hf/pocket-tts-farsi-v2` weights: CC-BY-NC-4.0, non-commercial
  use only.** Anything that ships or serves the pocket_v2 engine inherits
  that restriction; use OmniVoice for commercial use.
- pocket-tts (mallahyari fork of kyutai-labs/pocket-tts) code, including the
  vendored `pocket_normalize_fa.py`: MIT.
- `mehdi-hf/Homo-GE2PE-Persian-HF`: MIT (base byt5-small, Apache-2.0).
- `openai/whisper-large-v3-turbo`: MIT.
- `eustlb/higgs-audio-v2-tokenizer` (transformers port of Boson AI's Higgs
  Audio v2 tokenizer): its model card states no license; check upstream
  before redistributing.
