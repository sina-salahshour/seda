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
| STT | `openai/whisper-large-v3-turbo` via transformers, language `fa` | fp16 on GPU. |

VRAM is tight: load engines lazily, keep at most one TTS engine on GPU, and
expose `POST /unload`. Before training, ask Ollama to unload running models
(`GET :11434/api/ps`, then `POST /api/generate {"model": m, "keep_alive": 0}`)
and unload the sidecar's own inference models.

## Data directory

`$SEDA_DATA` (default `~/.local/share/seda`). The sidecar owns `profiles/`, the
Rust app owns `sessions/` and `settings.json`.

```
profiles/<profile_id>/profile.json
profiles/<profile_id>/samples/<sample_id>.wav     # 24 kHz mono 16-bit, loudness-normalised
profiles/<profile_id>/samples/<sample_id>.json
profiles/<profile_id>/runs/<run_id>/run.json      # + adapter/, logs, prepared data
sessions/<session_id>.json
settings.json
```

IDs: short lowercase random strings (e.g. 8 hex chars). Timestamps: RFC 3339 UTC.

## Sidecar HTTP API (FastAPI, `127.0.0.1:8765`, JSON unless noted)

Errors: non-2xx with `{"detail": "<message>"}`. Status codes: 400 invalid input, 404 unknown id, 409 busy (training queue full, or OmniVoice/GPU requested while training), 503 CUDA out of memory (sidecar unloads its models; retry), 500 other.

### Health
- `GET /health` → `{"ok": true, "device": "cuda"|"cpu", "gpu": {"name": str, "vram_total_mb": int, "vram_used_mb": int}|null, "engines": [{"id": "omnivoice"|"pocket_v2", "name": str, "loaded": bool}], "stt_loaded": bool, "busy": str|null}`
- `POST /unload` → `{"ok": true}` (frees all inference models)

### Profiles
Profile: `{"id", "name", "engine": "omnivoice"|"pocket_v2", "created_at", "active_run": run_id|null, "reference_sample": sample_id|null, "sample_count": int, "total_seconds": float}`
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
Run: `{"id", "profile_id", "engine", "status": "queued"|"running"|"done"|"failed"|"cancelled", "progress": 0..1, "step": int, "total_steps": int, "loss": float|null, "message": str, "created_at", "finished_at": str|null, "sample_count": int, "params": {...}}`
- `POST /profiles/{pid}/train` `{"steps"?: int, "from_run"?: run_id|null}` → Run.
  Runs one job at a time in the background (separate process) and trains on every current sample.
  With `from_run`, it continues from that adapter; otherwise it trains fresh from the base model.
  On success, the profile's `active_run` is set to the new run.
- `GET /profiles/{pid}/runs` → `[Run]` newest first
- `GET /profiles/{pid}/runs/{rid}` → Run (poll every ~1 s while running)
- `GET /profiles/{pid}/runs/{rid}/log` → `text/plain` (tail)
- `POST /profiles/{pid}/runs/{rid}/cancel` → Run
- `DELETE /profiles/{pid}/runs/{rid}` → `{"ok": true}`

### Synthesis / recognition
- `POST /tts` `{"profile_id", "text", "engine"?, "run_id"?: run_id|null, "speed"?: float}` → `audio/wav` (24 kHz mono 16-bit).
  - `engine` defaults to the profile's engine.
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
