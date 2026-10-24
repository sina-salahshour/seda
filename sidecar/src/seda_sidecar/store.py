"""On-disk store for profiles, samples and runs (see SPEC.md "Data directory").

All JSON writes are atomic (tmp + rename) and guarded by one process-wide lock.
"""
from __future__ import annotations

import json
import secrets
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .engines import DEFAULT_ENGINE, ENGINE_IDS

_lock = threading.RLock()

ENGINES = ENGINE_IDS  # derived from engines.REGISTRY


class NotFound(Exception):
    pass


def new_id() -> str:
    return secrets.token_hex(4)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{secrets.token_hex(2)}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- paths
def profile_dir(pid: str) -> Path:
    return config.PROFILES_DIR / pid


def samples_dir(pid: str) -> Path:
    return profile_dir(pid) / "samples"


def sample_wav(pid: str, sid: str) -> Path:
    return samples_dir(pid) / f"{sid}.wav"


def runs_dir(pid: str) -> Path:
    return profile_dir(pid) / "runs"


def run_dir(pid: str, rid: str) -> Path:
    return runs_dir(pid) / rid


# ---------------------------------------------------------------- profiles
def _raw_profile(pid: str) -> dict:
    p = profile_dir(pid) / "profile.json"
    if not p.is_file():
        raise NotFound(f"profile {pid} not found")
    return read_json(p)


def _decorate(prof: dict) -> dict:
    samples = list_samples(prof["id"])
    prof = dict(prof)
    prof["sample_count"] = len(samples)
    prof["total_seconds"] = float(round(sum(float(s["duration"]) for s in samples), 3))
    return prof


def get_profile(pid: str) -> dict:
    with _lock:
        return _decorate(_raw_profile(pid))


def list_profiles() -> list[dict]:
    with _lock:
        out = []
        if config.PROFILES_DIR.is_dir():
            for d in config.PROFILES_DIR.iterdir():
                if (d / "profile.json").is_file():
                    out.append(_decorate(read_json(d / "profile.json")))
        out.sort(key=lambda p: p.get("created_at", ""))
        return out


def create_profile(name: str, engine: str = DEFAULT_ENGINE) -> dict:
    with _lock:
        pid = new_id()
        prof = {
            "id": pid,
            "name": name,
            "engine": engine,
            "created_at": now(),
            "active_run": None,
            "reference_sample": None,
        }
        samples_dir(pid).mkdir(parents=True, exist_ok=True)
        runs_dir(pid).mkdir(parents=True, exist_ok=True)
        write_json(profile_dir(pid) / "profile.json", prof)
        return _decorate(prof)


def update_profile(pid: str, **fields: Any) -> dict:
    with _lock:
        prof = _raw_profile(pid)
        for k, v in fields.items():
            if k in ("name", "engine", "active_run", "reference_sample"):
                prof[k] = v
        write_json(profile_dir(pid) / "profile.json", prof)
        return _decorate(prof)


def delete_profile(pid: str) -> None:
    with _lock:
        _raw_profile(pid)
        shutil.rmtree(profile_dir(pid))


# ---------------------------------------------------------------- samples
def list_samples(pid: str) -> list[dict]:
    d = samples_dir(pid)
    if not d.is_dir():
        return []
    out = [read_json(p) for p in d.glob("*.json")]
    out.sort(key=lambda s: (s.get("created_at", ""), s.get("id", "")))
    return out


def get_sample(pid: str, sid: str) -> dict:
    p = samples_dir(pid) / f"{sid}.json"
    if not p.is_file():
        raise NotFound(f"sample {sid} not found")
    return read_json(p)


def save_sample(pid: str, sample: dict) -> dict:
    """Write samples/<id>.json. The caller has already written samples/<id>.wav."""
    with _lock:
        _raw_profile(pid)
        write_json(samples_dir(pid) / f"{sample['id']}.json", sample)
        return sample


def delete_sample(pid: str, sid: str) -> None:
    with _lock:
        get_sample(pid, sid)
        (samples_dir(pid) / f"{sid}.json").unlink(missing_ok=True)
        sample_wav(pid, sid).unlink(missing_ok=True)
        prof = _raw_profile(pid)
        if prof.get("reference_sample") == sid:
            prof["reference_sample"] = None
            write_json(profile_dir(pid) / "profile.json", prof)


# ---------------------------------------------------------------- runs
def list_runs(pid: str) -> list[dict]:
    _raw_profile(pid)
    out = []
    d = runs_dir(pid)
    if d.is_dir():
        for r in d.iterdir():
            if (r / "run.json").is_file():
                try:
                    out.append(read_json(r / "run.json"))
                except json.JSONDecodeError:
                    continue
    out.sort(key=lambda r: (r.get("created_at", ""), r["id"]), reverse=True)
    return out


def get_run(pid: str, rid: str) -> dict:
    p = run_dir(pid, rid) / "run.json"
    if not p.is_file():
        raise NotFound(f"run {rid} not found")
    return read_json(p)


def save_run(run: dict) -> dict:
    with _lock:
        write_json(run_dir(run["profile_id"], run["id"]) / "run.json", run)
        return run


def delete_run(pid: str, rid: str) -> None:
    with _lock:
        get_run(pid, rid)
        shutil.rmtree(run_dir(pid, rid))
        prof = _raw_profile(pid)
        if prof.get("active_run") == rid:
            prof["active_run"] = None
            write_json(profile_dir(pid) / "profile.json", prof)


# ---------------------------------------------------------------- engine view
@dataclass
class SampleRef:
    id: str
    text: str
    duration: float
    wav: Path  # 24 kHz mono 16-bit, loudness-normalised


@dataclass
class VoiceProfile:
    """What an engine needs to know about a profile."""

    id: str
    dir: Path
    engine: str
    reference_sample: str | None
    samples: list[SampleRef] = field(default_factory=list)

    def reference(self) -> SampleRef | None:
        for s in self.samples:
            if s.id == self.reference_sample:
                return s
        return None


def voice_profile(pid: str) -> VoiceProfile:
    prof = _raw_profile(pid)
    samples = [
        SampleRef(s["id"], s.get("text", ""), float(s["duration"]), sample_wav(pid, s["id"]))
        for s in list_samples(pid)
        if sample_wav(pid, s["id"]).is_file()
    ]
    return VoiceProfile(pid, profile_dir(pid), prof["engine"], prof.get("reference_sample"), samples)
