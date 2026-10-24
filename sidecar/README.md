# seda-sidecar

Python ML server for Seda (صدا): Persian TTS with voice cloning (three engines),
Whisper STT, per-profile LoRA training and a model manager (catalog, local model
folders, downloads). The Rust app talks to it over HTTP; the contract is
`../SPEC.md`.

| Role | Model | Device |
|---|---|---|
| Main TTS + LoRA training | `k2-fsa/OmniVoice` (+ `eustlb/higgs-audio-v2-tokenizer`) | GPU, fp16 inference / bf16 training |
| Fast TTS | `mehdi-hf/pocket-tts-farsi-v2` + G2P `mehdi-hf/Homo-GE2PE-Persian-HF` | CPU |
| Persian XTTS (engine `xtts`) | `MohammadJRanjbar/ParsVoice-XTTS` (gated) via `coqui-tts` | GPU fp32 |
| STT (default) | `openai/whisper-large-v3-turbo`, language `fa` | GPU fp16 (CPU while training) |
| STT (alternatives) | `openai/whisper-large-v3`, `BuzzASR/persian` | same |

Engines are registered in `engines/__init__.py` (`REGISTRY`: id, class, name,
uses_gpu, trainable, license, required catalog models, training defaults);
everything else (profile validation, `/health`, the training worker) derives
from it. Models are listed in `catalog.py`.

## Setup

Requires uv, ffmpeg and an NVIDIA driver recent enough for CUDA 12.8 (the
RTX 50xx / sm_120 needs torch built for cu128; `pyproject.toml` pins
`torch==2.8.0` from the cu128 index).

```sh
cd sidecar
uv sync                      # creates .venv with Python 3.12
# optional: pre-download the default models (~5 GB) into the HF cache
for m in k2-fsa/OmniVoice mehdi-hf/pocket-tts-farsi-v2 \
         mehdi-hf/Homo-GE2PE-Persian-HF openai/whisper-large-v3-turbo; do
  uv run hf download $m; done
```

Or download from the app (Models page → `POST /models/{id}/download`). Only
ParsVoice-XTTS is gated: accept its terms on huggingface.co and set a token
(`PATCH /models/config {"hf_token": "hf_..."}`; `kyutai/pocket-tts` is not used).
`coqui-tts` (idiap fork, for the xtts engine) is a regular dependency in
`pyproject.toml`; it resolves with torch 2.8.0 cu128 and transformers 5, so no
separate environment is needed.

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
| `SEDA_OMNIVOICE_MODEL` / `SEDA_POCKET_MODEL` / `SEDA_G2P_MODEL` / `SEDA_WHISPER_MODEL` | unset | force a repo id / local path, overriding `models.json` |
| `SEDA_XTTS_TRAIN_DEVICE` | `cpu` | device for computing XTTS latents in the training worker |
| `SEDA_EXTRA_CATALOG` | unset | **tests only**: JSON list of extra catalog entries (see `catalog.py`) |

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
| `engines/__init__.py` | engine registry (single source of truth for engine ids + metadata) |
| `engines/base.py` | `Engine`: `load / unload / synthesize(text, profile, run_dir, speed) -> float32 @ 24 kHz`, `train(job, run_dir, emit)` (in the worker), `artifact_ok(run_dir)` |
| `engines/omnivoice_engine.py`, `engines/omnivoice_train.py` | OmniVoice zero-shot cloning + PEFT LoRA adapters; LoRA training |
| `engines/pocket_engine.py` | pocket-tts-farsi-v2 pipeline + `prepare_pocket_reference` |
| `engines/xtts_engine.py` | ParsVoice-XTTS (coqui-tts) + cached conditioning latents |
| `catalog.py` | model catalog (repos, kinds, licenses, download patterns) |
| `models.py`, `download_worker.py` | model manager: `models.json`, resolution, model folders, downloads (child process) |
| `text_fa.py` | Persian TTS normalisation (digits → words, ZWNJ, ي/ك → ی/ک, punctuation) |
| `pocket_normalize_fa.py`, `pocket_latin.py` | model-specific normaliser (vendored from the fork, MIT) and Latin → Persian respelling |
| `audio.py` | ffmpeg import → 24 kHz mono, silence trim, loudness normalisation (-20 dBFS RMS, peak ≤ -1 dBFS), split > 30 s |
| `stt.py` | Whisper (`fa`, any Whisper-architecture model), long-form by chunking on silences |
| `training.py` / `train_worker.py` | job queue + training subprocess (dispatches to `Engine.train`) |
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
  OmniVoice, `pocket/` for pocket_v2, `xtts/latents.pt` for xtts) and fall back to zero-shot otherwise.
  A profile without samples → 400.
- Training: one job runs, at most one more waits (a third → 409). The worker is
  a separate process (`python -m seda_sidecar.train_worker <run_dir>`); its
  `@@PROGRESS {json}` lines are mirrored into `run.json` and everything goes to
  `train.log`. `cancel` kills the process group and returns the run once it is
  `cancelled`. On sidecar restart, stale queued/running runs become `failed`.
- While an OmniVoice training run owns the GPU (other engines' runs do not take it), GPU engines' `/tts` (omnivoice, xtts) → 409, pocket_v2 keeps
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

## Models

Catalog (`GET /models`): `omnivoice`, `higgs_tokenizer` (aux), `pocket_v2`,
`homo_ge2pe` (aux), `whisper_large_v3_turbo`, `whisper_large_v3`,
`buzzasr_persian`, `parsvoice_xtts`. `size_mb` comes from the hub
(`model_info(files_metadata=True)`, filtered by the entry's download patterns)
and is cached in `$SEDA_DATA/models_meta.json` (refreshed in the background at
start-up, at most weekly), so listing never waits on the network.

Resolution order of a catalog model: a matching dir in a model folder → the
complete HF cache snapshot → the repo id (downloaded on first use, as before).
`k2-fsa/OmniVoice` ships its own `audio_tokenizer/`, so `higgs_tokenizer` is
only needed for training with an OmniVoice dir that lacks it.

Model folders (`model_dirs`): each dir and its immediate subdirs. A dir named
like a catalog repo (`OmniVoice`, `k2-fsa--OmniVoice`, `k2-fsa_OmniVoice`)
counts as that catalog model; otherwise it is a local model `local:<path>` if
detected (`config.json` `model_type: "whisper"` → stt; `"model": "xtts"` or
`model.pth` + `vocab.json` → xtts; `model_type: "omnivoice"` → omnivoice;
`model.yaml` + `tokenizer_ph.model` → pocket_v2). Seda never deletes files in
model folders.

Downloads: `python -m seda_sidecar.download_worker` in its own process group,
`snapshot_download` into the normal HF cache with the stored token, Xet
disabled so `blobs/*.incomplete` grows while downloading; progress =
bytes of the repo's files on disk / total. Logs: `$SEDA_DATA/downloads/<id>.log`.

`$SEDA_DATA/models.json` (0600): `stt_model`, `model_dirs`, `engine_models`,
`hf_token` (also exported as `HF_TOKEN` to the sidecar and its children).

## Engine: xtts (ParsVoice-XTTS)

`Xtts(config).load_checkpoint(dir)` from `config.json` + `model.pth` +
`vocab.json`. coqui's tokenizer has no Persian cleaner, so text goes through
`normalize_fa_text` and the tokenizer's `encode` is wrapped to skip its cleaners
for `fa` (tag `[fa]` if the vocab has it, else `[ar]` with a warning). Text is
split into ≤ 180-char pieces (sentences, then clauses), joined with 0.2 s
pauses; output is 24 kHz float32. Sampling settings come from the checkpoint's
config (temperature, top_k/p, repetition penalty); `speed` uses XTTS's own
length scaling. "Training" computes `gpt_cond_latent` + `speaker_embedding`
from the reference sample and the cleanest 3–15 s clips (≤ 12 clips, ≤ 45 s)
on the CPU in the worker and saves `runs/<rid>/xtts/latents.pt`; without a run,
latents come from ≤ 4 clips / 20 s on the fly (LRU). The pipeline was verified
with a randomly initialised tiny XTTS (coqui-tts 0.28 + transformers 5.2);
the real checkpoint is gated and has not been run here.

## Training

`POST /profiles/{pid}/train` runs the profile engine's `train()` in the worker.
Only omnivoice is `trainable` (steps / `from_run`); pocket_v2 and xtts "prepare
the voice" (reference clip / latents, CPU, no GPU hand-over) and ignore steps /
`from_run`. Success = the engine's `artifact_ok(run_dir)`. `run.json` records
`models: {role: "<id or repo>@<revision or path>"}`.

### OmniVoice LoRA

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
- `openai/whisper-large-v3-turbo`: MIT. `openai/whisper-large-v3`: Apache-2.0.
- `BuzzASR/persian` (Whisper large-v3 fine-tuned for Persian): MIT.
- **`MohammadJRanjbar/ParsVoice-XTTS`: CPML (Coqui Public Model License),
  non-commercial use only.** Its card metadata says MIT, but it is a fine-tune
  of Coqui XTTS-v2, whose weights are CPML; treat it as non-commercial. The repo
  is gated (accept the terms on huggingface.co). coqui-tts code: MPL-2.0.
- Higgs Audio v2 tokenizer (`eustlb/higgs-audio-v2-tokenizer`, and the copy in
  `k2-fsa/OmniVoice/audio_tokenizer/`): Boson Higgs Audio 2 Community License
  (based on the Llama 3 community license; commercial use with its conditions).
