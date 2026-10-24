# Seda (صدا) — Persian voice cloning + voice chat

Desktop app that clones the user's Persian voice and talks with local Ollama
models and Claude Code using it.

```
app/      Rust, gpui-kit 0.8 desktop UI           (crate name: seda)
sidecar/  Python (uv, py3.12, torch cu128) ML server (package: seda_sidecar)
```

Hardware target: RTX 5070 Laptop (8 GB VRAM, Blackwell sm_120 → torch built for
cu128 or newer), 30 GB RAM, Arch Linux + Hyprland (Wayland). Ollama runs
locally on :11434 (model `qwen3.8`, 17 GB, partly offloaded to RAM).

## Models

| Role | Model | Notes |
|---|---|---|
| Main TTS + training | `k2-fsa/OmniVoice` (Apache-2.0) | Persian (`fa`) in training data (366 h). Zero-shot cloning from 3–10 s ref + ref_text. LoRA fine-tuning per profile (`docs/lora_finetuning.md`). |
| Fast / CPU TTS | `mehdi-hf/pocket-tts-farsi-v2` (CC-BY-NC) | Needs G2P `mehdi-hf/Homo-GE2PE-Persian-HF` + `normalize_fa.py`; prompt ≤ 5 s; chunk text to 9–16 tokens; join chunks with 0.25 s silence; retry runaway chunks. Installed from `git+https://github.com/mallahyari/pocket-tts@main`. "Training" for this engine = choosing/trimming the best ≤5 s reference clip from the profile's samples. |
| Persian XTTS | `MohammadJRanjbar/ParsVoice-XTTS` (CPML, non-commercial; gated) | Engine `xtts` via `coqui-tts`. "Training" = caching speaker conditioning latents from the profile's best clips. Needs a Hugging Face token + accepted terms. |
| STT | `openai/whisper-large-v3-turbo` via transformers, language `fa` (default) | fp16 on GPU. Alternatives in the catalog: `openai/whisper-large-v3`, `BuzzASR/persian` (MIT, best Persian WER). Only Whisper-architecture models. |

The sidecar's model catalog, model folders and downloads are described under
"Models" in the API section; the engine list comes from `sidecar/src/seda_sidecar/engines/__init__.py`.

VRAM is tight: load engines lazily, keep at most one TTS engine on GPU, and
expose `POST /unload`. Before training, ask Ollama to unload running models
(`GET :11434/api/ps`, then `POST /api/generate {"model": m, "keep_alive": 0}`)
and unload the sidecar's own inference models.

## Data directory

`$SEDA_DATA` (default `~/.local/share/seda`). The sidecar owns `profiles/`,
`models.json` and its caches; the Rust app owns `sessions/` and `settings.json`.

```
profiles/<profile_id>/profile.json
profiles/<profile_id>/samples/<sample_id>.wav     # 24 kHz mono 16-bit, loudness-normalised
profiles/<profile_id>/samples/<sample_id>.json
profiles/<profile_id>/runs/<run_id>/run.json      # + adapter/ | pocket/ | xtts/latents.pt, logs, prepared data
models.json                                       # sidecar model config (mode 0600, holds the HF token)
models_meta.json, downloads/, cache/              # sidecar caches (repo file sizes, download logs, generated configs)
sessions/<session_id>.json
settings.json
```

IDs: short lowercase random strings (e.g. 8 hex chars). Timestamps: RFC 3339 UTC.

## Sidecar HTTP API (FastAPI, `127.0.0.1:8765`, JSON unless noted)

Errors: non-2xx with `{"detail": "<message>"}`. Status codes: 400 invalid input, 404 unknown id, 409 busy (training queue full, or OmniVoice/GPU requested while training), 503 CUDA out of memory (sidecar unloads its models; retry), 500 other.

### Health
- `GET /health` → `{"ok": true, "device": "cuda"|"cpu", "gpu": {"name": str, "vram_total_mb": int, "vram_used_mb": int}|null, "engines": [Engine], "stt_loaded": bool, "stt_model": str, "busy": str|null}`
  - Engine: `{"id": "omnivoice"|"pocket_v2"|"xtts", "name": str, "loaded": bool, "uses_gpu": bool, "trainable": bool, "ready": bool, "license": str, "commercial": bool, "model": str}`.
    `trainable` = step-based fine-tuning (`steps` / `from_run` apply; only omnivoice). `ready` = every model the engine needs is on disk. `model` = model id in use (catalog id or `local:<path>`).
  - `stt_model`: the STT model id in use (`SEDA_WHISPER_MODEL` wins when set).
- `POST /unload` → `{"ok": true}` (frees all inference models)

### Profiles
Profile: `{"id", "name", "engine": "omnivoice"|"pocket_v2"|"xtts", "created_at", "active_run": run_id|null, "reference_sample": sample_id|null, "sample_count": int, "total_seconds": float}`
- `GET /profiles` → `[Profile]`
- `POST /profiles` `{"name", "engine"?}` → Profile (engine defaults to `omnivoice`)
- `GET /profiles/{pid}` → Profile
- `PATCH /profiles/{pid}` `{"name"?, "engine"?, "active_run"?, "reference_sample"?}` → Profile (`active_run: null` = zero-shot, no adapter)
- `DELETE /profiles/{pid}` → `{"ok": true}`

### Samples
Sample: `{"id", "text", "duration": float, "created_at", "source": "recorded"|"imported"}`
- `GET /profiles/{pid}/samples` → `[Sample]`
- `POST /profiles/{pid}/samples` `{"wav_path": abs path to any audio file ffmpeg reads, "text"?: str, "source"?: str}` → Sample.
  The sidecar converts to 24 kHz mono, trims silence, and normalises loudness. Files longer than 30 s are split on silences into several samples; the response is then the first sample, and `GET` lists all of them.
  If `text` is missing or empty, it is transcribed with Whisper (`fa`).
- `PATCH /profiles/{pid}/samples/{sid}` `{"text"}` → Sample
- `DELETE /profiles/{pid}/samples/{sid}` → `{"ok": true}`
- `GET /profiles/{pid}/samples/{sid}/audio` → `audio/wav`
- `GET /prompts?n=10` → `["<Persian sentence>", ...]`: a phonetically varied set of sentences for the user to read aloud (bundle ~200 sentences in the sidecar).

### Training ("train" / "retrain")
Run: `{"id", "profile_id", "engine", "status": "queued"|"running"|"done"|"failed"|"cancelled", "progress": 0..1, "step": int, "total_steps": int, "loss": float|null, "message": str, "created_at", "finished_at": str|null, "sample_count": int, "params": {...}, "models": {"<role>": "<model id or repo>@<revision or path>"}}`
(`models` is absent on runs created before it existed; roles: `tts` plus the engine's auxiliary model ids, e.g. `homo_ge2pe`.)

What a run produces depends on the profile's engine: omnivoice → LoRA adapter (`adapter/`);
pocket_v2 → best ≤ 5 s reference clip (`pocket/`); xtts → speaker latents (`xtts/latents.pt`).
For engines with `trainable: false` (pocket_v2, xtts) a run just "prepares the voice": `steps` / `from_run` are accepted and ignored, `total_steps` is 1.
- `POST /profiles/{pid}/train` `{"steps"?: int, "from_run"?: run_id|null}` → Run.
  Runs one job at a time in the background (separate process) and trains on every current sample.
  With `from_run`, it continues from that adapter; otherwise it trains fresh from the base model.
  On success, the profile's `active_run` is set to the new run.
- `GET /profiles/{pid}/runs` → `[Run]` newest first
- `GET /profiles/{pid}/runs/{rid}` → Run (poll every ~1 s while running)
- `GET /profiles/{pid}/runs/{rid}/log` → `text/plain` (tail)
- `POST /profiles/{pid}/runs/{rid}/cancel` → Run
- `DELETE /profiles/{pid}/runs/{rid}` → `{"ok": true}`

### Models
Model: `{"id": str, "kind": "tts"|"stt"|"aux", "engine": str|null, "name": str, "description": str, "repo": str|null, "path": str|null, "source": "catalog"|"local", "license": str, "commercial": bool, "gated": bool, "size_mb": int|null, "status": "missing"|"downloading"|"ready"|"error", "downloaded_mb": float, "error": str|null, "in_use": bool}`

Catalog ids: `omnivoice`, `higgs_tokenizer` (aux), `pocket_v2`, `homo_ge2pe` (aux), `whisper_large_v3_turbo` (default STT), `whisper_large_v3`, `buzzasr_persian`, `parsvoice_xtts` (gated).
Local models (found in the configured model folders) have id `local:<abs path>`, `repo: null`, `license: "unknown"`.
`engine` of an aux model is the engine that needs it; `null` for STT. `size_mb` is `null` until the repo's file sizes have been fetched once (cached on disk).

Where a catalog model is loaded from: (a) a matching dir in a model folder, (b) the complete HF cache snapshot, (c) the repo id (downloaded on first use). `path` is the local dir of (a)/(b), else `null`.
Model folders: each configured dir and each of its immediate subdirs; a dir named like a catalog repo (`OmniVoice`, `k2-fsa--OmniVoice`, `k2-fsa_OmniVoice`) *is* that catalog model, otherwise it is listed as local when its kind is detected (Whisper `config.json` → stt; XTTS / OmniVoice / pocket-tts → tts), else skipped.

- `GET /models` → `[Model]` (catalog first, then local; no network per call)
- `POST /models/{id}/download` → Model (`status: "downloading"`; poll `GET /models` for `downloaded_mb`). 409 if already downloading, 400 for local models, 404 unknown id. A gated repo without a token, or 401/403, ends in `status: "error"` with a Persian + English `error`.
- `POST /models/{id}/cancel` → Model (partial files stay in the HF cache; a later download resumes)
- `DELETE /models/{id}` → `{"ok": true}`: deletes the model's HF-cache copy only (unloading it first); never touches files in model folders. 400 for local models.
- `GET /models/config` → `{"stt_model": str, "model_dirs": [str], "engine_models": {engine_id: model id}, "hf_token_set": bool}`
- `PATCH /models/config` `{"stt_model"?, "model_dirs"?, "engine_models"?, "hf_token"?}` → same shape as GET.
  - `stt_model`: a catalog STT id or `local:<abs path>` of a Whisper model (else 400). Changing it unloads STT.
  - `model_dirs`: absolute paths of existing dirs (else 400).
  - `engine_models`: replaces the whole map; a missing engine means its catalog default. Values: a catalog TTS id of that engine or `local:<path>` of a matching model. Changed engines are unloaded.
  - `hf_token`: `""` clears. Stored in `$SEDA_DATA/models.json` (mode 0600), never returned.
- `{id}` may contain `/` (local ids): send it percent-encoded (`local%3A%2Fhome%2F...`) or raw; both are routed.

### Synthesis / recognition
- `POST /tts` `{"profile_id", "text", "engine"?, "run_id"?: run_id|null, "speed"?: float}` → `audio/wav` (24 kHz mono 16-bit).
  - `engine` defaults to the profile's engine (`omnivoice`, `pocket_v2`, `xtts`).
  - A missing gated model (xtts without a token) → 400 with the gated message.
  - `run_id` is omitted → use the profile's `active_run`; explicit `null` → zero-shot.
  - Text is Persian script and may contain digits, Latin words, and punctuation; the sidecar normalises it.
  - The client sends one sentence or short paragraph per call.
- `POST /stt` `{"wav_path"}` → `{"text"}` (Persian)

## Rust app (gpui-kit 0.8)

- Spawns the sidecar if `/health` fails (`uv run --project <repo>/sidecar seda-sidecar`), shows its status, and kills the child on quit.
  - Settings: sidecar URL, sidecar autostart, Ollama URL, default system prompt, `claude` binary path, Claude permission mode.
- Persian text everywhere. Bundle the Vazirmatn font (OFL) and use it as the UI font, falling back for Latin. The layout must render RTL text correctly.
- **Voices page:** profile list + create; profile detail:
  - Samples list with play, edit transcript, and delete.
  - Record (cpal → WAV temp file → `POST samples`); while recording, a prompt sentence from `/prompts` is shown to read aloud.
  - Import audio files.
  - Train / Retrain buttons, with a run list showing progress, loss, status, set-active, and delete.
  - Engine select.
  - "Try it": text box + speak, with an A/B choice of run.
- **Chats page:** session list + new session.
  - Backend is either Ollama (model from `GET /api/tags`) or Claude Code (working directory picker).
  - Each session has a voice profile and an auto-speak toggle.
  - Messages render as markdown.
  - Input: textarea, plus push-to-talk (record → `/stt` → fills the input, optionally auto-sends).
  - Assistant replies stream. Completed sentences are queued to `/tts` and played in order (rodio) while streaming continues.
  - A stop button cancels both generation and playback.
  - Ollama: `POST /api/chat` with `stream: true`, NDJSON.
  - Claude Code: spawn `claude -p <msg> --output-format stream-json --verbose --include-partial-messages [--resume <claude_session_id>] [--permission-mode <mode>]` in the session's cwd.
    Read `session_id` from the init event and keep it in the session; stream text deltas; show tool-use events compactly.
  - Default system prompt (Ollama) / appended prompt (Claude, `--append-system-prompt`): reply in natural spoken Persian, short sentences, no markdown tables when voice is on.
