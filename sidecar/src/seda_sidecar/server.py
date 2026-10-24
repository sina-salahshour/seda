"""Seda sidecar HTTP API (see SPEC.md). Entry point: `seda-sidecar` -> main()."""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from . import audio, config, manager, models, prompts, store
from .engines import DEFAULT_ENGINE

log = logging.getLogger("seda.server")

SOURCES = ("recorded", "imported")
ACTIVE_RUN_STATES = ("queued", "running")


# ---------------------------------------------------------------- training (owned by another module)
def _training():
    """Import the training module lazily so the server still starts without it."""
    try:
        from . import training
    except Exception as e:  # pragma: no cover - only while training.py is missing/broken
        log.exception("training module unavailable")
        raise HTTPException(503, f"training is unavailable: {e}") from e
    return training


def _is_training_busy(exc: Exception) -> bool:
    try:
        from . import training

        return isinstance(exc, training.Busy)
    except Exception:
        return False


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    config.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        models.startup()
    except Exception:
        log.exception("models.startup() failed")
    try:
        from . import training

        training.recover()
    except Exception:
        log.exception("training.recover() failed")
    log.info("seda sidecar ready on %s:%s, data in %s", config.HOST, config.PORT, config.DATA_DIR)
    try:
        yield
    finally:
        try:
            from . import training

            training.shutdown()
        except Exception:
            log.exception("training.shutdown() failed")
        try:
            models.shutdown()
        except Exception:
            log.exception("models.shutdown() failed")


app = FastAPI(title="Seda sidecar", version="0.1.0", lifespan=lifespan)


def _err(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


@app.exception_handler(RequestValidationError)
async def _validation_handler(_: Request, exc: RequestValidationError):
    parts = []
    for e in exc.errors():
        loc = ".".join(str(p) for p in e.get("loc", ()) if p not in ("body", "query", "path"))
        parts.append(f"{loc}: {e.get('msg')}" if loc else str(e.get("msg")))
    return _err(400, "; ".join(parts) or "invalid request")


@app.exception_handler(store.NotFound)
async def _notfound_handler(_: Request, exc: store.NotFound):
    return _err(404, str(exc))


@app.exception_handler(models.ModelError)
async def _model_error_handler(_: Request, exc: models.ModelError):
    return _err(400, str(exc))


@app.exception_handler(models.ModelConflict)
async def _model_conflict_handler(_: Request, exc: models.ModelConflict):
    return _err(409, str(exc))


@app.exception_handler(models.ModelNotFound)
async def _model_notfound_handler(_: Request, exc: models.ModelNotFound):
    return _err(404, str(exc))


@app.exception_handler(manager.GpuBusy)
async def _gpubusy_handler(_: Request, exc: manager.GpuBusy):
    return _err(409, str(exc))


@app.exception_handler(audio.AudioError)
async def _audio_handler(_: Request, exc: audio.AudioError):
    return _err(400, str(exc))


@app.exception_handler(Exception)
async def _generic_handler(_: Request, exc: Exception):
    # Starlette routes unhandled exceptions here (and still logs the traceback).
    if _is_training_busy(exc):
        return _err(409, str(exc) or "training queue is full")
    if type(exc).__name__ == "OutOfMemoryError":
        # Another process (Ollama, a second sidecar, ...) took the VRAM. Drop our
        # models so the next request starts from a clean slate.
        try:
            manager.unload_all()
        except Exception:
            log.exception("unload after OOM failed")
        return _err(503, "GPU out of memory (is another process using the GPU?); models were unloaded, retry")
    return _err(500, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- health
_gpu_static: dict[str, Any] = {}


def _cuda_available() -> bool:
    if "cuda" not in _gpu_static:
        try:
            import torch

            _gpu_static["cuda"] = bool(torch.cuda.is_available())
        except Exception:
            _gpu_static["cuda"] = False
    return _gpu_static["cuda"]


def _gpu_info() -> dict | None:
    if not _cuda_available():
        return None
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=True,
            ).stdout.strip().splitlines()[0]
            name, total, used = (s.strip() for s in out.rsplit(",", 2))
            return {"name": name, "vram_total_mb": int(float(total)), "vram_used_mb": int(float(used))}
        except Exception:
            log.debug("nvidia-smi failed", exc_info=True)
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        return {
            "name": torch.cuda.get_device_name(0),
            "vram_total_mb": total // 2**20,
            "vram_used_mb": (total - free) // 2**20,
        }
    except Exception:
        return None


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": "cuda" if _cuda_available() else "cpu",
        "gpu": _gpu_info(),
        "engines": manager.engines_info(),
        "stt_loaded": manager.stt_loaded(),
        "stt_model": models.stt_model_id(),
        "busy": manager.busy(),
    }


@app.post("/unload")
def unload():
    manager.unload_all()
    return {"ok": True}


# ---------------------------------------------------------------- profiles
class ProfileCreate(BaseModel):
    name: str
    engine: str = DEFAULT_ENGINE


class ProfilePatch(BaseModel):
    name: str | None = None
    engine: str | None = None
    active_run: str | None = None
    reference_sample: str | None = None


def _check_engine(engine: str) -> str:
    if engine not in store.ENGINES:
        raise HTTPException(400, f"unknown engine {engine!r} (expected one of {', '.join(store.ENGINES)})")
    return engine


def _check_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise HTTPException(400, "name must not be empty")
    return name


@app.get("/profiles")
def list_profiles():
    return store.list_profiles()


@app.post("/profiles")
def create_profile(body: ProfileCreate):
    return store.create_profile(_check_name(body.name), _check_engine(body.engine))


@app.get("/profiles/{pid}")
def get_profile(pid: str):
    return store.get_profile(pid)


@app.patch("/profiles/{pid}")
def patch_profile(pid: str, body: ProfilePatch):
    store.get_profile(pid)
    fields: dict[str, Any] = {}
    sent = body.model_fields_set
    if "name" in sent:
        if body.name is None:
            raise HTTPException(400, "name must not be null")
        fields["name"] = _check_name(body.name)
    if "engine" in sent:
        if body.engine is None:
            raise HTTPException(400, "engine must not be null")
        fields["engine"] = _check_engine(body.engine)
    if "active_run" in sent:
        if body.active_run is not None:
            try:
                run = store.get_run(pid, body.active_run)
            except store.NotFound as e:
                raise HTTPException(400, str(e)) from e
            if run.get("status") != "done":
                raise HTTPException(400, f"run {body.active_run} is {run.get('status')}, not done")
        fields["active_run"] = body.active_run
    if "reference_sample" in sent:
        if body.reference_sample is not None:
            try:
                store.get_sample(pid, body.reference_sample)
            except store.NotFound as e:
                raise HTTPException(400, str(e)) from e
        fields["reference_sample"] = body.reference_sample
    return store.update_profile(pid, **fields)


@app.delete("/profiles/{pid}")
def delete_profile(pid: str):
    store.get_profile(pid)
    active = [r for r in store.list_runs(pid) if r.get("status") in ACTIVE_RUN_STATES]
    if active:
        tr = _training()
        for r in active:
            try:
                tr.delete_run(pid, r["id"])
            except Exception:
                log.exception("could not stop run %s before deleting profile %s", r["id"], pid)
    store.delete_profile(pid)
    return {"ok": True}


# ---------------------------------------------------------------- samples
class SampleCreate(BaseModel):
    wav_path: str
    text: str | None = None
    source: str | None = None


class SamplePatch(BaseModel):
    text: str


@app.get("/profiles/{pid}/samples")
def list_samples(pid: str):
    store.get_profile(pid)
    return store.list_samples(pid)


@app.post("/profiles/{pid}/samples")
def add_sample(pid: str, body: SampleCreate):
    store.get_profile(pid)
    source = body.source or "imported"
    if source not in SOURCES:
        raise HTTPException(400, f"source must be one of {', '.join(SOURCES)}")
    text = (body.text or "").strip()
    chunks = audio.prepare_import(body.wav_path)
    if len(chunks) > 1 and text:
        # One transcript cannot be divided between chunks reliably; transcribe each.
        log.info("%s was split into %d chunks; ignoring the given text and transcribing", body.wav_path, len(chunks))
        text = ""
    texts: list[str]
    if text:
        texts = [text]
    else:
        t0 = time.monotonic()
        with manager.stt(busy_label="transcribe") as stt:
            texts = [stt.transcribe(audio.to_16k(c)).strip() for c in chunks]
        log.info("transcribed %d chunk(s) in %.1fs", len(chunks), time.monotonic() - t0)

    # Sorted ids keep the chunks in order (samples are listed by created_at, id).
    ids = sorted(store.new_id() for _ in chunks)
    created = store.now()
    saved = []
    for sid, chunk, t in zip(ids, chunks, texts):
        audio.write_wav(store.sample_wav(pid, sid), chunk, audio.SR)
        sample = {
            "id": sid,
            "text": t,
            "duration": audio.duration(chunk),
            "created_at": created,
            "source": source,
        }
        try:
            saved.append(store.save_sample(pid, sample))
        except store.NotFound:
            store.sample_wav(pid, sid).unlink(missing_ok=True)  # profile deleted meanwhile
            raise
    return saved[0]


@app.patch("/profiles/{pid}/samples/{sid}")
def patch_sample(pid: str, sid: str, body: SamplePatch):
    store.get_profile(pid)
    sample = store.get_sample(pid, sid)
    sample["text"] = body.text.strip()
    return store.save_sample(pid, sample)


@app.delete("/profiles/{pid}/samples/{sid}")
def delete_sample(pid: str, sid: str):
    store.get_profile(pid)
    store.delete_sample(pid, sid)
    return {"ok": True}


@app.get("/profiles/{pid}/samples/{sid}/audio")
def sample_audio(pid: str, sid: str):
    store.get_profile(pid)
    store.get_sample(pid, sid)
    path = store.sample_wav(pid, sid)
    if not path.is_file():
        raise store.NotFound(f"audio for sample {sid} not found")
    return FileResponse(path, media_type="audio/wav", filename=f"{sid}.wav")


@app.get("/prompts")
def get_prompts(n: int = Query(10, ge=1, le=500)):
    return prompts.sample(n)


# ---------------------------------------------------------------- training
class TrainRequest(BaseModel):
    steps: int | None = None
    from_run: str | None = None


@app.post("/profiles/{pid}/train")
def train(pid: str, body: TrainRequest | None = None):
    body = body or TrainRequest()
    prof = store.get_profile(pid)
    from .engines import REGISTRY

    sp = REGISTRY.get(prof["engine"])
    if sp is not None and not sp.trainable:
        body = TrainRequest()  # "prepare voice" engines ignore steps / from_run
    if body.steps is not None and body.steps <= 0:
        raise HTTPException(400, "steps must be positive")
    tr = _training()
    try:
        return tr.start_run(pid, body.steps, body.from_run)
    except store.NotFound:
        raise
    except tr.Busy as e:
        raise HTTPException(409, str(e) or "a training run is already queued") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/profiles/{pid}/runs")
def list_runs(pid: str):
    return store.list_runs(pid)


@app.get("/profiles/{pid}/runs/{rid}")
def get_run(pid: str, rid: str):
    store.get_profile(pid)
    return store.get_run(pid, rid)


@app.get("/profiles/{pid}/runs/{rid}/log", response_class=PlainTextResponse)
def run_log(pid: str, rid: str):
    store.get_profile(pid)
    store.get_run(pid, rid)
    return PlainTextResponse(_training().log_tail(pid, rid))


@app.post("/profiles/{pid}/runs/{rid}/cancel")
def cancel_run(pid: str, rid: str):
    store.get_profile(pid)
    store.get_run(pid, rid)
    return _training().cancel_run(pid, rid)


@app.delete("/profiles/{pid}/runs/{rid}")
def delete_run(pid: str, rid: str):
    store.get_profile(pid)
    store.get_run(pid, rid)
    _training().delete_run(pid, rid)
    return {"ok": True}


# ---------------------------------------------------------------- models
class ModelsConfigPatch(BaseModel):
    stt_model: str | None = None
    model_dirs: list[str] | None = None
    engine_models: dict[str, str | None] | None = None
    hf_token: str | None = None


@app.get("/models")
def list_models():
    return models.list_models()


@app.get("/models/config")
def get_models_config():
    return models.public_config()


@app.patch("/models/config")
def patch_models_config(body: ModelsConfigPatch):
    patch = {k: getattr(body, k) for k in body.model_fields_set}
    cfg, changed = models.update_config(patch)
    if "stt" in changed:
        manager.reset_stt()
    for c in changed:
        if c.startswith("engine:"):
            manager.unload_engine(c.split(":", 1)[1])
    return cfg


@app.post("/models/{mid:path}/download")
def download_model(mid: str):
    return models.start_download(mid)


@app.post("/models/{mid:path}/cancel")
def cancel_model_download(mid: str):
    return models.cancel_download(mid)


@app.delete("/models/{mid:path}")
def delete_model(mid: str):
    models.delete_model(mid)
    return {"ok": True}


# ---------------------------------------------------------------- synthesis / recognition
class TTSRequest(BaseModel):
    profile_id: str
    text: str
    engine: str | None = None
    run_id: str | None = None
    speed: float | None = None


@app.post("/tts")
def tts(body: TTSRequest):
    prof = store.get_profile(body.profile_id)
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "text must not be empty")
    eid = _check_engine(body.engine or prof["engine"])
    speed = 1.0 if body.speed is None else float(body.speed)
    if not 0.5 <= speed <= 2.0:
        raise HTTPException(400, "speed must be between 0.5 and 2.0")

    # run_id omitted -> profile's active run; explicit null -> zero-shot.
    rid = body.run_id if "run_id" in body.model_fields_set else prof.get("active_run")
    run_dir = None
    if rid:
        run = store.get_run(prof["id"], rid)  # 404 if an explicit run id does not exist
        # Engines pick their own artefacts from the run dir (adapter/ for omnivoice,
        # pocket/ for pocket_v2) and fall back to zero-shot when theirs is missing.
        if run.get("status") == "done":
            run_dir = store.run_dir(prof["id"], rid)
        else:
            log.info("run %s (%s, %s) not usable with %s; zero-shot", rid, run.get("status"), run.get("engine"), eid)

    vp = store.voice_profile(prof["id"])
    if not vp.samples:
        raise HTTPException(400, "the profile has no samples")
    t0 = time.monotonic()
    with manager.tts_engine(eid, busy_label="tts") as eng:
        wav = eng.synthesize(text, vp, run_dir, speed)
    secs = len(wav) / audio.SR
    log.info("tts %s run=%s: %d chars -> %.2fs audio in %.2fs", eid, rid if run_dir else None, len(text), secs, time.monotonic() - t0)
    return Response(audio.wav_bytes(wav, audio.SR), media_type="audio/wav")


class STTRequest(BaseModel):
    wav_path: str


@app.post("/stt")
def stt(body: STTRequest):
    x = audio.load_16k(body.wav_path)
    t0 = time.monotonic()
    with manager.stt(busy_label="stt") as m:
        text = m.transcribe(x)
    log.info("stt %.2fs audio in %.2fs", len(x) / audio.STT_SR, time.monotonic() - t0)
    return {"text": text.strip()}


# ---------------------------------------------------------------- entry point
def main() -> None:
    import os

    import uvicorn

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")


if __name__ == "__main__":
    main()
