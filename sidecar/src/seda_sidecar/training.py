"""Training job manager: per-engine "training" runs in a subprocess.

What a run produces depends on the profile's engine (``engines.REGISTRY`` /
``Engine.train``): an OmniVoice LoRA adapter, a pocket_v2 reference clip, or
XTTS voice latents. Success = the engine's ``artifact_ok(run_dir)``.

One job runs at a time, at most one more waits in the queue. The child is
``sys.executable -m seda_sidecar.train_worker <run_dir>`` in its own process
group; its stdout goes to run_dir/train.log and its ``@@PROGRESS {json}`` lines
are mirrored into run.json.

OmniVoice's default step count (``default_steps``) scales with the amount of audio:

    steps = clamp(round(150 + 12 * minutes), 200, 1000)

One optimizer step sees ``batch_tokens * grad_accum`` = 3072 * 2 tokens (25 audio
tokens/s, so roughly 3 min of speech plus text; length bucketing leaves some
partial batches). Measured: 3.3 min of audio ~ 3 steps per epoch. So 5 min -> 210
steps, 30 min -> 510, 60 min -> 870 steps, i.e. roughly 40-60 passes over the data
at every size. At ~1 s/step on an RTX 5070 laptop (8 GB; peak ~3.5 GB allocated)
that is ~4 min for 5 min of audio and ~15 min for an hour, plus ~20 s start-up.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from . import config, engines, manager, models, store

log = logging.getLogger("seda.training")

PROGRESS_PREFIX = "@@PROGRESS "
MAX_QUEUED = 1
MIN_STEPS, MAX_STEPS = 1, 20000

# OmniVoice LoRA defaults (kept for compatibility; per-engine: engines.default_params).
DEFAULT_PARAMS = engines.default_params("omnivoice")


class Busy(Exception):
    """Another job is running and the queue is full (HTTP 409)."""


_lock = threading.RLock()
_queue: deque[tuple[str, str]] = deque()
_current: dict | None = None  # {"pid", "rid", "proc", "cancel": bool}
_thread: threading.Thread | None = None
_wake = threading.Event()
_stop = False


def default_steps(total_seconds: float, engine: str = "omnivoice") -> int:
    return engines.default_steps(engine, total_seconds)


def run_models(eid: str) -> tuple[dict[str, str], dict[str, str]]:
    """({role: "<model id or repo>@<revision or path>"}, {role: path or repo}) for a run."""
    ids = models.engine_model_ids(eid)
    paths = {role: models.resolve(mid) for role, mid in ids.items()}
    if eid == "omnivoice":
        tts = Path(paths["tts"])
        if tts.is_dir() and not (tts / "audio_tokenizer").is_dir():
            ids["audio_tokenizer"] = "higgs_tokenizer"
            paths["audio_tokenizer"] = models.resolve("higgs_tokenizer")
    return {role: models.model_ref(mid) for role, mid in ids.items()}, paths


def _usable_samples(pid: str) -> list[dict]:
    out = []
    for s in store.list_samples(pid):
        wav = store.sample_wav(pid, s["id"])
        if wav.is_file() and (s.get("text") or "").strip():
            out.append({"id": s["id"], "wav": str(wav), "text": s["text"], "duration": float(s["duration"])})
    return out


def _update(pid: str, rid: str, **fields) -> dict | None:
    with store._lock:
        try:
            run = store.get_run(pid, rid)
        except (store.NotFound, json.JSONDecodeError):
            return None
        run.update(fields)
        return store.save_run(run)


# ---------------------------------------------------------------- public API
def start_run(pid: str, steps: int | None = None, from_run: str | None = None) -> dict:
    prof = store.get_profile(pid)  # raises NotFound
    eid = prof["engine"]
    if eid not in engines.REGISTRY:
        raise ValueError(f"unknown engine {eid!r}")
    sp = engines.spec(eid)
    if not sp.trainable:
        # "prepare voice" engines: steps / from_run do not apply and are ignored
        steps, from_run = None, None
    samples = _usable_samples(pid)
    if not samples:
        raise ValueError("profile has no samples with transcripts to train on")
    eng = manager.engine(eid)
    from_run_dir = None
    if from_run:
        prev = store.get_run(pid, from_run)  # raises NotFound
        prev_dir = store.run_dir(pid, from_run)
        if prev.get("status") != "done" or not eng.artifact_ok(prev_dir):
            raise ValueError(f"run {from_run} has no finished {eid} artefact to continue from")
        from_run_dir = str(prev_dir)
    run_model_refs, model_paths = run_models(eid)
    total = sum(s["duration"] for s in samples)
    if steps is None:
        steps = default_steps(total, eid)
    steps = int(steps)
    if not MIN_STEPS <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")

    with _lock:
        if _current is not None and len(_queue) >= MAX_QUEUED:
            raise Busy("a training run is already in progress and another is queued")
        rid = store.new_id()
        params = dict(engines.default_params(eid), steps=steps, from_run=from_run, total_seconds=round(total, 2))
        run = {
            "id": rid,
            "profile_id": pid,
            "engine": eid,
            "status": "queued",
            "progress": 0.0,
            "step": 0,
            "total_steps": steps,
            "loss": None,
            "message": "queued",
            "created_at": store.now(),
            "finished_at": None,
            "sample_count": len(samples),
            "params": params,
            "models": run_model_refs,
        }
        rdir = store.run_dir(pid, rid)
        rdir.mkdir(parents=True, exist_ok=True)
        job = {"profile_id": pid, "engine": eid, "samples": samples, "params": params,
               "from_run_dir": from_run_dir, "model_paths": model_paths}
        store.write_json(rdir / "job.json", job)
        store.save_run(run)
        _queue.append((pid, rid))
        _ensure_thread()
        _wake.set()
        return run


def cancel_run(pid: str, rid: str) -> dict:
    run = store.get_run(pid, rid)
    with _lock:
        if (pid, rid) in _queue:
            _queue.remove((pid, rid))
            return _update(pid, rid, status="cancelled", message="cancelled", finished_at=store.now()) or run
        cur = _current
        if cur is not None and cur["pid"] == pid and cur["rid"] == rid:
            cur["cancel"] = True
            _kill(cur.get("proc"))
            _update(pid, rid, message="cancelling")
            waiting = True
        else:
            waiting = False
    if waiting:
        # Give the job thread a moment to reap the worker so the caller sees the
        # final status ("cancelled") rather than "running".
        deadline = time.time() + 10
        while time.time() < deadline:
            with _lock:
                cur = _current
                if cur is None or cur["pid"] != pid or cur["rid"] != rid:
                    break
            time.sleep(0.1)
        return store.get_run(pid, rid)
    if run.get("status") in ("queued", "running"):  # orphaned
        return _update(pid, rid, status="cancelled", message="cancelled", finished_at=store.now()) or run
    return run


def delete_run(pid: str, rid: str) -> None:
    store.get_run(pid, rid)
    cancel_run(pid, rid)
    deadline = time.time() + 15
    while time.time() < deadline:
        with _lock:
            cur = _current
            if cur is None or cur["rid"] != rid or cur["pid"] != pid:
                break
        time.sleep(0.1)
    store.delete_run(pid, rid)


def log_tail(pid: str, rid: str, max_bytes: int = 20000) -> str:
    store.get_run(pid, rid)
    p = store.run_dir(pid, rid) / "train.log"
    if not p.is_file():
        return ""
    with open(p, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read()
    return data.decode("utf-8", errors="replace")


def active() -> dict | None:
    """{"pid", "rid"} of the running job, if any."""
    with _lock:
        return {"pid": _current["pid"], "rid": _current["rid"]} if _current else None


def recover() -> None:
    """At startup: runs left queued/running by a previous sidecar are failed."""
    for prof in store.list_profiles():
        for run in store.list_runs(prof["id"]):
            if run.get("status") in ("queued", "running"):
                _update(prof["id"], run["id"], status="failed", message="sidecar restarted",
                        finished_at=store.now())


def shutdown() -> None:
    global _stop
    with _lock:
        _stop = True
        for pid, rid in list(_queue):
            _update(pid, rid, status="cancelled", message="sidecar stopped", finished_at=store.now())
        _queue.clear()
        if _current is not None:
            _current["cancel"] = True
            _kill(_current.get("proc"), wait=True)
    _wake.set()


# ---------------------------------------------------------------- internals
def _kill(proc: subprocess.Popen | None, wait: bool = False) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    def _reap():
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    if wait:
        _reap()
    else:
        threading.Thread(target=_reap, daemon=True).start()


def _ensure_thread() -> None:
    global _thread, _stop
    if _thread is None or not _thread.is_alive():
        _stop = False
        _thread = threading.Thread(target=_loop, name="seda-training", daemon=True)
        _thread.start()


def _loop() -> None:
    global _current
    while not _stop:
        with _lock:
            job = _queue.popleft() if _queue else None
            if job is not None:
                _current = {"pid": job[0], "rid": job[1], "proc": None, "cancel": False}
        if job is None:
            _wake.wait(5)
            _wake.clear()
            continue
        try:
            _run_job(*job)
        except Exception as e:  # noqa: BLE001
            log.exception("training job crashed")
            _update(job[0], job[1], status="failed", message=f"error: {e}", finished_at=store.now())
        finally:
            with _lock:
                _current = None
            manager.TRAINING.clear()


def _ollama_unload() -> None:
    try:
        import httpx

        with httpx.Client(base_url=config.OLLAMA_URL, timeout=5) as c:
            ps = c.get("/api/ps").json()
            for m in ps.get("models", []) or []:
                name = m.get("name") or m.get("model")
                if name:
                    log.info("asking ollama to unload %s", name)
                    c.post("/api/generate", json={"model": name, "keep_alive": 0}, timeout=30)
    except Exception as e:  # noqa: BLE001 - ollama is optional
        log.info("ollama unload skipped: %s", e)


def _run_job(pid: str, rid: str) -> None:
    try:
        run = store.get_run(pid, rid)
    except store.NotFound:
        return
    if run.get("status") != "queued":
        return
    rdir = store.run_dir(pid, rid)
    eid = run.get("engine") or "omnivoice"
    sp = engines.spec(eid) if eid in engines.REGISTRY else engines.spec("omnivoice")
    if sp.train_gpu:
        _update(pid, rid, status="running", message="freeing GPU memory")
        manager.TRAINING.set()  # blocks new GPU inference before we unload
        manager.unload_all()
        _ollama_unload()
    else:
        _update(pid, rid, status="running", message="starting")

    with _lock:
        if _current is None or _current["cancel"]:
            _update(pid, rid, status="cancelled", message="cancelled", finished_at=store.now())
            return
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONUNBUFFERED"] = "1"
        env["SEDA_DATA"] = str(config.DATA_DIR)
        logf = open(rdir / "train.log", "ab", buffering=0)
        proc = subprocess.Popen(
            [sys.executable, "-m", "seda_sidecar.train_worker", str(rdir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        _current["proc"] = proc
    _update(pid, rid, message="starting")
    last_write = 0.0
    pending: dict = {}
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace")
            if line.startswith(PROGRESS_PREFIX):
                try:
                    ev = json.loads(line[len(PROGRESS_PREFIX):])
                except json.JSONDecodeError:
                    logf.write(raw)
                    continue
                for k in ("progress", "step", "total_steps", "loss", "message"):
                    if k in ev:
                        pending[k] = ev[k]
                stats = {k: ev[k] for k in ("sec_per_step", "peak_vram_mb") if k in ev}
                if stats:
                    pending["_stats"] = stats
                now = time.time()
                if now - last_write >= 0.5 or ev.get("stage") in ("save", "done"):
                    _flush(pid, rid, pending)
                    pending = {}
                    last_write = now
            else:
                logf.write(raw)
        proc.wait()
    finally:
        if pending:
            _flush(pid, rid, pending)
        logf.close()

    rc = proc.returncode
    with _lock:
        cancelled = bool(_current and _current["cancel"])
    artifact_ok = manager.engine(sp.id).artifact_ok(rdir)
    if cancelled:
        _update(pid, rid, status="cancelled", message="cancelled", finished_at=store.now())
    elif rc == 0 and artifact_ok:
        _update(pid, rid, status="done", progress=1.0, message="done", finished_at=store.now())
        try:
            store.update_profile(pid, active_run=rid)
        except store.NotFound:
            pass
    else:
        tail = log_tail(pid, rid, 4000).strip().splitlines()
        reason = next((ln for ln in reversed(tail) if ln.strip()), f"exit code {rc}")
        if "OutOfMemoryError" in "\n".join(tail) or "CUDA out of memory" in "\n".join(tail):
            reason = "out of GPU memory (close other GPU apps and retry)"
        _update(pid, rid, status="failed", message=f"failed: {reason[-300:]}", finished_at=store.now())


def _flush(pid: str, rid: str, pending: dict) -> None:
    fields = {k: v for k, v in pending.items() if not k.startswith("_")}
    if "_stats" in pending:
        run = store.get_run(pid, rid) if store.run_dir(pid, rid).is_dir() else None
        if run is not None:
            params = dict(run.get("params") or {})
            params.update(pending["_stats"])
            fields["params"] = params
    if fields:
        _update(pid, rid, **fields)
