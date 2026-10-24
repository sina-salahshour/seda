"""Model manager: catalog status, local model folders, downloads, models.json.

Where a catalog model is loaded from (``resolve``):
  (a) a matching model directory inside a user model folder (``model_dirs``),
  (b) the complete HF cache snapshot,
  (c) the bare repo id (the engine then downloads it on first use, as before).
``SEDA_OMNIVOICE_MODEL`` / ``SEDA_POCKET_MODEL`` / ``SEDA_G2P_MODEL`` (and
``SEDA_WHISPER_MODEL`` for STT) still override everything when set.

Model folders: every configured dir and each of its immediate subdirs is
inspected. A dir whose name matches a catalog repo ("OmniVoice",
"k2-fsa--OmniVoice", "k2-fsa_OmniVoice", ...) *is* that catalog model;
otherwise it is listed as a local model ("local:<abs path>") if its kind can be
detected (Whisper → stt, XTTS / OmniVoice / pocket-tts → tts), else skipped.

Downloads run ``python -m seda_sidecar.download_worker`` in its own process
group (so cancel can kill it) into the normal HF cache. Progress is polled from
disk: bytes of the repo's files already in the snapshot + ``blobs/*.incomplete``.

Config ($SEDA_DATA/models.json, mode 0600):
  {"stt_model": catalog id | "local:<abs path>", "model_dirs": [abs paths],
   "engine_models": {engine_id: model id}, "hf_token": str | null}
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import catalog, config
from .catalog import CatalogModel
from .engines import REGISTRY, spec

log = logging.getLogger("seda.models")

CONFIG_PATH = config.DATA_DIR / "models.json"
META_PATH = config.DATA_DIR / "models_meta.json"
DOWNLOAD_LOGS = config.DATA_DIR / "downloads"
META_MAX_AGE = 7 * 24 * 3600
LOCAL_PREFIX = "local:"
ERROR_PREFIX = "@@ERROR "

_lock = threading.RLock()
_NON_ESSENTIAL = ("README*", "*.md", ".gitattributes", "LICENSE*", "NOTICE*", "*.ipynb", "samples/*")
_WEIGHT_GLOBS = ("*.safetensors", "*.pth", "*.bin", "*.pt", "*.ckpt")

GATED_MSG = (
    "این مدل محدود (gated) است: در huggingface.co شرایط آن را بپذیرید و توکن Hugging Face را در تنظیمات مدل‌ها وارد کنید. "
    "/ Gated model: accept its terms on huggingface.co and set a Hugging Face token in the model settings."
)


class ModelError(ValueError):
    """Invalid model request (HTTP 400)."""


class ModelConflict(Exception):
    """E.g. a download already running (HTTP 409)."""


class ModelNotFound(Exception):
    """Unknown model id (HTTP 404)."""


# ====================================================================== config
def _default_config() -> dict:
    return {"stt_model": catalog.DEFAULT_STT, "model_dirs": [], "engine_models": {}, "hf_token": None}


_cfg: dict | None = None
_env_token_set_by_us = False


def load_config() -> dict:
    global _cfg
    with _lock:
        if _cfg is None:
            cfg = _default_config()
            try:
                if CONFIG_PATH.is_file():
                    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        for k in cfg:
                            if k in raw and raw[k] is not None:
                                cfg[k] = raw[k]
                        if raw.get("hf_token") is None:
                            cfg["hf_token"] = None
            except Exception as e:  # noqa: BLE001
                log.warning("cannot read %s: %s; using defaults", CONFIG_PATH, e)
            if not isinstance(cfg["model_dirs"], list):
                cfg["model_dirs"] = []
            if not isinstance(cfg["engine_models"], dict):
                cfg["engine_models"] = {}
            _cfg = cfg
            _apply_token(cfg.get("hf_token"))
        return _cfg


def _save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(f".json.tmp{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.chmod(tmp, 0o600)
    tmp.replace(CONFIG_PATH)


def _apply_token(token: str | None) -> None:
    """Expose the stored token to huggingface_hub (this process and its children)."""
    global _env_token_set_by_us
    if token:
        os.environ["HF_TOKEN"] = token
        _env_token_set_by_us = True
    elif _env_token_set_by_us:
        os.environ.pop("HF_TOKEN", None)
        _env_token_set_by_us = False


def hf_token() -> str | None:
    """Stored token, else whatever huggingface_hub finds (HF_TOKEN env, `hf auth login`)."""
    tok = load_config().get("hf_token")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # noqa: BLE001
        return None


def public_config() -> dict:
    cfg = load_config()
    return {
        "stt_model": stt_model_id(),
        "model_dirs": list(cfg["model_dirs"]),
        "engine_models": dict(cfg["engine_models"]),
        "hf_token_set": bool(cfg.get("hf_token")),
    }


def update_config(patch: dict) -> tuple[dict, set[str]]:
    """Validate and apply a PATCH. Returns (public config, changed) where changed holds
    "stt" and/or "engine:<id>" for things that must be unloaded."""
    with _lock:
        before = load_config()
        try:
            return _update_config(dict(before), patch)
        except Exception:
            _set_cfg(before)
            _apply_token(before.get("hf_token"))
            raise


def _update_config(cfg: dict, patch: dict) -> tuple[dict, set[str]]:
    changed: set[str] = set()
    if "model_dirs" in patch:
        dirs = patch["model_dirs"]
        if dirs is None:
            dirs = []
        if not isinstance(dirs, list) or not all(isinstance(d, str) for d in dirs):
            raise ModelError("model_dirs must be a list of paths")
        out: list[str] = []
        for d in dirs:
            p = Path(d).expanduser()
            if not p.is_absolute():
                raise ModelError(f"model_dirs: {d!r} is not an absolute path")
            if not p.is_dir():
                raise ModelError(f"model_dirs: {d} does not exist or is not a directory")
            s = str(p.resolve())
            if s not in out:
                out.append(s)
        old_eff = {eid: _engine_resolved_key(eid) for eid in REGISTRY}
        old_stt = _stt_resolved_key()
        cfg["model_dirs"] = out
        _set_cfg(cfg)
        for eid in REGISTRY:
            if _engine_resolved_key(eid) != old_eff[eid]:
                changed.add(f"engine:{eid}")
        if _stt_resolved_key() != old_stt:
            changed.add("stt")
    if "stt_model" in patch:
        sid = patch["stt_model"] or catalog.DEFAULT_STT
        _validate_stt(sid)
        if sid != cfg.get("stt_model"):
            changed.add("stt")
        cfg["stt_model"] = sid
    if "engine_models" in patch:
        em = patch["engine_models"] or {}
        if not isinstance(em, dict):
            raise ModelError("engine_models must be an object {engine_id: model id}")
        # The map is replaced as a whole; a missing engine (or null / "") = catalog default.
        old = dict(cfg.get("engine_models") or {})
        new: dict[str, str] = {}
        for eid, mid in em.items():
            if eid not in REGISTRY:
                raise ModelError(f"engine_models: unknown engine {eid!r}")
            if mid in (None, ""):
                continue
            _validate_engine_model(eid, mid)
            new[eid] = mid
        for eid in set(old) | set(new):
            if old.get(eid) != new.get(eid):
                changed.add(f"engine:{eid}")
        cfg["engine_models"] = new
    if "hf_token" in patch:
        tok = patch["hf_token"]
        cfg["hf_token"] = (tok.strip() or None) if isinstance(tok, str) else None
        _apply_token(cfg["hf_token"])
    _set_cfg(cfg)
    _save_config(cfg)
    return public_config(), changed


def _set_cfg(cfg: dict) -> None:
    global _cfg
    _cfg = cfg
    _scan_cache.clear()


def _validate_stt(sid: str) -> None:
    if not isinstance(sid, str):
        raise ModelError("stt_model must be a string")
    m = catalog.get(sid)
    if m is not None:
        if m.kind != "stt":
            raise ModelError(f"{sid} is not a speech-recognition (Whisper) model")
        return
    if sid.startswith(LOCAL_PREFIX):
        p = Path(sid[len(LOCAL_PREFIX):])
        if not p.is_absolute() or not p.is_dir():
            raise ModelError(f"stt_model: {p} does not exist")
        kind = detect(p)
        if kind is None or kind[0] != "stt":
            raise ModelError(f"stt_model: {p} is not a Whisper model (config.json model_type must be \"whisper\")")
        return
    raise ModelError(f"unknown stt_model {sid!r} (expected a catalog id or \"local:<abs path>\")")


def _validate_engine_model(eid: str, mid: str) -> None:
    if not isinstance(mid, str):
        raise ModelError("engine_models values must be model ids")
    m = catalog.get(mid)
    if m is not None:
        if m.kind != "tts" or m.engine != eid:
            raise ModelError(f"{mid} is not a TTS model for engine {eid}")
        return
    if mid.startswith(LOCAL_PREFIX):
        p = Path(mid[len(LOCAL_PREFIX):])
        if not p.is_absolute() or not p.is_dir():
            raise ModelError(f"engine_models: {p} does not exist")
        kind = detect(p)
        if kind is None or kind != ("tts", eid):
            raise ModelError(f"engine_models: {p} is not a {eid} model")
        return
    raise ModelError(f"unknown model {mid!r}")


# ====================================================================== HF cache + metadata
def _hub_cache() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def _repo_cache_dir(repo: str) -> Path:
    return _hub_cache() / ("models--" + repo.replace("/", "--"))


_meta: dict[str, dict] | None = None


def _load_meta() -> dict[str, dict]:
    global _meta
    if _meta is None:
        try:
            _meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.is_file() else {}
        except Exception:  # noqa: BLE001
            _meta = {}
    return _meta


def _save_meta() -> None:
    try:
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = META_PATH.with_suffix(f".json.tmp{os.getpid()}")
        tmp.write_text(json.dumps(_load_meta(), indent=1), encoding="utf-8")
        tmp.replace(META_PATH)
    except Exception as e:  # noqa: BLE001
        log.warning("cannot write %s: %s", META_PATH, e)


def _allowed(m: CatalogModel, name: str) -> bool:
    if m.allow_patterns is None:
        return True
    return any(fnmatch.fnmatch(name, p) for p in m.allow_patterns)


def fetch_meta(m: CatalogModel, force: bool = False) -> dict | None:
    """{"sha", "files": {name: size}, "gated", "fetched"} for the files this model downloads."""
    with _lock:
        cur = _load_meta().get(m.id)
    if cur and not force and time.time() - cur.get("fetched", 0) < META_MAX_AGE and cur.get("repo") == m.repo:
        return cur
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(m.repo, revision=m.revision, files_metadata=True, token=hf_token())
        files = {s.rfilename: int(s.size or 0) for s in info.siblings or [] if _allowed(m, s.rfilename)}
        new = {"repo": m.repo, "sha": info.sha, "files": files, "gated": bool(info.gated), "fetched": time.time()}
    except Exception as e:  # noqa: BLE001 - offline is fine
        log.info("model info for %s unavailable: %s", m.repo, e)
        return cur
    with _lock:
        _load_meta()[m.id] = new
        _save_meta()
    return new


def refresh_meta_async() -> None:
    def run():
        for m in catalog.CATALOG.values():
            fetch_meta(m)

    threading.Thread(target=run, name="seda-model-meta", daemon=True).start()


def _meta_for(m: CatalogModel) -> dict | None:
    with _lock:
        cur = _load_meta().get(m.id)
    return cur if cur and cur.get("repo") == m.repo else None


def _essential(name: str) -> bool:
    return not any(fnmatch.fnmatch(name, p) for p in _NON_ESSENTIAL)


def _snapshot_dir(m: CatalogModel) -> Path | None:
    d = _repo_cache_dir(m.repo)
    ref = d / "refs" / m.revision
    sha = None
    if ref.is_file():
        sha = ref.read_text().strip()
    elif len(m.revision) == 40:
        sha = m.revision
    else:
        meta = _meta_for(m)
        sha = meta.get("sha") if meta else None
    if not sha:
        return None
    snap = d / "snapshots" / sha
    return snap if snap.is_dir() else None


def _cache_complete(m: CatalogModel) -> Path | None:
    snap = _snapshot_dir(m)
    if snap is None:
        return None
    meta = _meta_for(m)
    if meta and meta.get("files") and meta.get("sha") == snap.name:
        need = [f for f in meta["files"] if _essential(f)]
        return snap if all((snap / f).exists() for f in need) else None
    # No (matching) file list: accept a snapshot that has a config and weights.
    has_cfg = (snap / "config.json").exists() or (snap / "model.yaml").exists()
    has_w = any(any(snap.glob(g)) for g in _WEIGHT_GLOBS)
    return snap if has_cfg and has_w else None


def _cached_bytes(m: CatalogModel) -> int:
    """Bytes of this model already on disk in the HF cache (finished files + partial blobs)."""
    d = _repo_cache_dir(m.repo)
    if not d.is_dir():
        return 0
    total = 0
    snap = _snapshot_dir(m)
    meta = _meta_for(m)
    if snap is not None:
        names = list(meta["files"]) if meta and meta.get("files") else None
        if names is None:
            names = [str(p.relative_to(snap)) for p in snap.rglob("*") if p.is_file()]
        for n in names:
            p = snap / n
            try:
                if p.exists():
                    total += p.stat().st_size
            except OSError:
                pass
    blobs = d / "blobs"
    if blobs.is_dir():
        for p in blobs.glob("*.incomplete"):
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _total_bytes(m: CatalogModel) -> int | None:
    meta = _meta_for(m)
    if meta and meta.get("files"):
        return int(sum(meta["files"].values()))
    return None


# ====================================================================== model folders
def _read_json(p: Path) -> dict | None:
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        return None


def detect(d: Path) -> tuple[str, str | None] | None:
    """(kind, engine) of a model directory, or None if it is not a model we can use."""
    cfgp = d / "config.json"
    cfg = _read_json(cfgp) if cfgp.is_file() else None
    if cfg is not None:
        mt = str(cfg.get("model_type") or "").lower()
        archs = [str(a) for a in cfg.get("architectures") or []]
        if mt == "whisper" or any(a.startswith("Whisper") for a in archs):
            return ("stt", None)
        if str(cfg.get("model") or "").lower() == "xtts":
            return ("tts", "xtts")
        if mt == "omnivoice" or "OmniVoice" in archs:
            return ("tts", "omnivoice")
        if (d / "model.pth").is_file() and (d / "vocab.json").is_file() and (
            "model_args" in cfg or "gpt" in json.dumps(cfg)[:20000]
        ):
            return ("tts", "xtts")
    if (d / "model.yaml").is_file() and (d / "tokenizer_ph.model").is_file():
        return ("tts", "pocket_v2")
    return None


def _looks_like_model(d: Path) -> bool:
    if (d / "config.json").is_file() or (d / "model.yaml").is_file():
        return True
    return any(any(d.glob(g)) for g in _WEIGHT_GLOBS)


def _name_keys(m: CatalogModel) -> set[str]:
    org, _, name = m.repo.partition("/")
    return {k.lower() for k in (name, f"{org}--{name}", f"{org}_{name}", f"{org}__{name}", m.id)}


def _dir_size(d: Path, max_depth: int = 4) -> int:
    total = 0
    base = len(d.parts)
    for root, dirs, files in os.walk(d):
        if len(Path(root).parts) - base >= max_depth:
            dirs[:] = []
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


_scan_cache: dict[str, Any] = {}


def _scan() -> tuple[dict[str, Path], list[tuple[Path, str, str | None]]]:
    """(catalog id -> folder path, [(local dir, kind, engine)]) over the configured model_dirs."""
    dirs = list(load_config()["model_dirs"])
    key = []
    for d in dirs:
        try:
            key.append((d, os.stat(d).st_mtime_ns))
        except OSError:
            key.append((d, 0))
    hit = _scan_cache.get("scan")
    if hit is not None and hit[0] == key and time.time() - hit[1] < 5:
        return hit[2]
    by_name: dict[str, str] = {}
    for m in catalog.CATALOG.values():
        for k in _name_keys(m):
            by_name.setdefault(k, m.id)
    matched: dict[str, Path] = {}
    local: list[tuple[Path, str, str | None]] = []
    seen: set[str] = set()
    for d in dirs:
        root = Path(d)
        if not root.is_dir():
            continue
        cands = [root]
        try:
            cands += sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            pass
        for c in cands:
            s = str(c.resolve())
            if s in seen:
                continue
            seen.add(s)
            mid = by_name.get(c.name.lower())
            if mid is not None and _looks_like_model(c):
                matched.setdefault(mid, Path(s))
                continue
            kind = detect(c)
            if kind is not None:
                local.append((Path(s), kind[0], kind[1]))
    res = (matched, local)
    _scan_cache["scan"] = (key, time.time(), res)
    return res


# ====================================================================== resolution
def _env_override(m: CatalogModel) -> str | None:
    return os.environ.get(m.env) if m.env and os.environ.get(m.env) else None


def local_path(mid: str) -> Path | None:
    """Local directory for a model id (folder or complete HF cache snapshot), or None."""
    if mid.startswith(LOCAL_PREFIX):
        p = Path(mid[len(LOCAL_PREFIX):])
        return p if p.is_dir() else None
    m = catalog.get(mid)
    if m is None:
        return None
    env = _env_override(m)
    if env:
        return Path(env) if Path(env).is_dir() else None
    folder = _scan()[0].get(mid)
    if folder is not None:
        return folder
    return _cache_complete(m)


def resolve(mid: str) -> str:
    """Where to load a model from: a local dir, else the repo id (downloaded on first use)."""
    if mid.startswith(LOCAL_PREFIX):
        p = Path(mid[len(LOCAL_PREFIX):])
        if not p.is_dir():
            raise ModelError(f"local model {p} does not exist")
        return str(p)
    m = catalog.get(mid)
    if m is None:
        raise ModelError(f"unknown model {mid!r}")
    env = _env_override(m)
    if env:
        return env
    p = local_path(mid)
    return str(p) if p is not None else m.repo


def model_ref(mid: str) -> str:
    """"<model id or repo>@<revision or path>" for run.json "models"."""
    if mid.startswith(LOCAL_PREFIX):
        return f"{mid}@{mid[len(LOCAL_PREFIX):]}"
    m = catalog.get(mid)
    if m is None:
        return mid
    env = _env_override(m)
    if env:
        return f"{mid}@{env}"
    folder = _scan()[0].get(mid)
    if folder is not None:
        return f"{mid}@{folder}"
    snap = _cache_complete(m)
    if snap is not None:
        return f"{m.repo}@{snap.name}"
    return f"{m.repo}@{m.revision}"


def engine_model_id(eid: str) -> str:
    return load_config().get("engine_models", {}).get(eid) or spec(eid).models[0]


def engine_model_ids(eid: str) -> dict[str, str]:
    """{role: model id}: "tts" = the engine's main model, then its auxiliary models by id."""
    out = {"tts": engine_model_id(eid)}
    for aux in spec(eid).models[1:]:
        out[aux] = aux
    return out


def engine_paths(eid: str) -> dict[str, str]:
    return {role: resolve(mid) for role, mid in engine_model_ids(eid).items()}


def engine_ready(eid: str) -> bool:
    return all(local_path(mid) is not None for mid in engine_model_ids(eid).values())


def _engine_resolved_key(eid: str) -> tuple:
    try:
        return tuple(sorted(engine_paths(eid).items()))
    except ModelError:
        return ()


def stt_model_id() -> str:
    return os.environ.get("SEDA_WHISPER_MODEL") or load_config().get("stt_model") or catalog.DEFAULT_STT


def stt_path() -> str:
    sid = stt_model_id()
    if sid.startswith(LOCAL_PREFIX) or catalog.get(sid) is not None:
        return resolve(sid)
    return sid  # SEDA_WHISPER_MODEL: a repo id or path, used as-is


def _stt_resolved_key() -> str:
    try:
        return stt_path()
    except ModelError:
        return ""


def stt_generate_kwargs() -> dict:
    m = catalog.get(stt_model_id())
    return dict(m.generate_kwargs) if m is not None else {}


# ====================================================================== downloads
_downloads: dict[str, dict] = {}  # id -> {"proc", "cancel", "thread"}
_errors: dict[str, str] = {}


def _downloading(mid: str) -> bool:
    d = _downloads.get(mid)
    return d is not None and not d.get("finished")


def start_download(mid: str) -> dict:
    m = _catalog_or_raise(mid)
    with _lock:
        if _downloading(mid):
            raise ModelConflict(f"{mid} is already downloading")
        _errors.pop(mid, None)
        if m.gated and not hf_token():
            _errors[mid] = GATED_MSG
            return model_info(mid)
        job = {"proc": None, "cancel": False, "finished": False, "started": time.time()}
        _downloads[mid] = job
        t = threading.Thread(target=_download_thread, args=(m, job), name=f"seda-dl-{mid}", daemon=True)
        job["thread"] = t
        t.start()
    return model_info(mid)


def _download_thread(m: CatalogModel, job: dict) -> None:
    rc = None
    logp = DOWNLOAD_LOGS / f"{m.id}.log"
    try:
        fetch_meta(m)
        DOWNLOAD_LOGS.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        tok = hf_token()
        if tok:
            env["HF_TOKEN"] = tok
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        # Plain HTTP downloads write blobs/*.incomplete as they go (Xet only writes the
        # file at the end), which is what the progress poll measures.
        env["HF_HUB_DISABLE_XET"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        src = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        arg = json.dumps({"repo": m.repo, "revision": m.revision,
                          "allow_patterns": list(m.allow_patterns) if m.allow_patterns else None})
        with _lock:
            if job["cancel"]:
                return
            with open(logp, "wb") as logf:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "seda_sidecar.download_worker", arg],
                    stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    env=env, start_new_session=True,
                )
            job["proc"] = proc
        log.info("downloading %s (%s), pid %d", m.id, m.repo, proc.pid)
        rc = proc.wait()
    except Exception as e:  # noqa: BLE001
        log.exception("download of %s failed to start", m.id)
        _errors[m.id] = f"{type(e).__name__}: {e}"
    finally:
        with _lock:
            job["finished"] = True
            _scan_cache.clear()
            if job["cancel"]:
                log.info("download of %s cancelled", m.id)
            elif rc == 0:
                _errors.pop(m.id, None)
                log.info("download of %s finished", m.id)
            elif rc is not None:
                _errors[m.id] = _download_error(logp, rc)
                log.warning("download of %s failed: %s", m.id, _errors[m.id])


def _download_error(logp: Path, rc: int) -> str:
    try:
        lines = logp.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for ln in reversed(lines):
        if ln.startswith(ERROR_PREFIX):
            try:
                e = json.loads(ln[len(ERROR_PREFIX):])
            except json.JSONDecodeError:
                break
            if e.get("gated") or e.get("status") in (401, 403):
                return GATED_MSG + f" ({e.get('message', '')[:200]})"
            if e.get("type") == "RepositoryNotFoundError":
                return f"مخزن پیدا نشد / repository not found: {e.get('message', '')[:300]}"
            if e.get("type") in ("LocalEntryNotFoundError", "ConnectError", "ConnectionError"):
                return f"خطای شبکه / network error: {e.get('message', '')[:300]}"
            return f"{e.get('type')}: {e.get('message', '')[:400]}"
    tail = next((ln for ln in reversed(lines) if ln.strip()), "")
    return f"download failed (exit code {rc}): {tail[-300:]}"


def cancel_download(mid: str) -> dict:
    _catalog_or_raise(mid)
    with _lock:
        job = _downloads.get(mid)
        if job is None or job.get("finished"):
            return model_info(mid)
        job["cancel"] = True
        proc = job.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
    t = job.get("thread")
    if t is not None:
        t.join(timeout=5)
    return model_info(mid)


def delete_model(mid: str) -> None:
    m = _catalog_or_raise(mid)
    if _downloading(mid):
        cancel_download(mid)
    from . import manager

    manager.unload_model(mid)
    d = _repo_cache_dir(m.repo)
    if d.is_dir():
        log.info("deleting HF cache copy of %s: %s", mid, d)
        shutil.rmtree(d)
    with _lock:
        _errors.pop(mid, None)
        _downloads.pop(mid, None)
        _scan_cache.clear()


def _catalog_or_raise(mid: str) -> CatalogModel:
    m = catalog.get(mid)
    if m is not None:
        return m
    if mid.startswith(LOCAL_PREFIX):
        raise ModelError("local models live in your own folders; Seda does not download or delete them")
    raise ModelNotFound(f"model {mid} not found")


# ====================================================================== listing
def _in_use() -> set[str]:
    try:
        from . import manager

        return manager.models_in_use()
    except Exception:  # noqa: BLE001
        return set()


def _mb(b: int | float | None) -> float:
    return round((b or 0) / 2**20, 1)


def _catalog_entry(m: CatalogModel, matched: dict[str, Path], in_use: set[str]) -> dict:
    meta = _meta_for(m)
    total = _total_bytes(m)
    env = _env_override(m)
    path: Path | None = None
    if env:
        path = Path(env) if Path(env).is_dir() else None
    if path is None:
        path = matched.get(m.id) or _cache_complete(m)
    downloading = _downloading(m.id)
    err = _errors.get(m.id)
    if downloading:
        status = "downloading"
    elif path is not None:
        status = "ready"
    elif err:
        status = "error"
    else:
        status = "missing"
    if status == "ready":
        if path is not None and path == matched.get(m.id):
            done = _dir_size(path)
            total = total or done
        else:
            done = total if total is not None else _cached_bytes(m)
    else:
        done = _cached_bytes(m)
    gated = bool(m.gated or (meta or {}).get("gated"))
    return {
        "id": m.id,
        "kind": m.kind,
        "engine": m.engine,
        "name": m.name,
        "description": m.description,
        "repo": m.repo,
        "path": str(path) if path is not None else None,
        "source": "catalog",
        "license": m.license,
        "commercial": m.commercial,
        "gated": gated,
        "size_mb": int(round(total / 2**20)) if total else None,
        "status": status,
        "downloaded_mb": _mb(done),
        "error": err if status == "error" else None,
        "in_use": m.id in in_use,
    }


_size_cache: dict[str, tuple[int, int]] = {}


def _local_entry(p: Path, kind: str, engine: str | None, in_use: set[str]) -> dict:
    try:
        mt = p.stat().st_mtime_ns
    except OSError:
        mt = 0
    hit = _size_cache.get(str(p))
    if hit is None or hit[0] != mt:
        hit = (mt, _dir_size(p))
        _size_cache[str(p)] = hit
    size = hit[1]
    mid = LOCAL_PREFIX + str(p)
    what = {"stt": "Whisper (STT)", "tts": f"TTS ({engine})"}.get(kind, kind)
    return {
        "id": mid,
        "kind": kind,
        "engine": engine,
        "name": p.name,
        "description": f"مدل محلی / local model: {what}",
        "repo": None,
        "path": str(p),
        "source": "local",
        "license": "unknown",
        "commercial": False,
        "gated": False,
        "size_mb": int(round(size / 2**20)),
        "status": "ready",
        "downloaded_mb": _mb(size),
        "error": None,
        "in_use": mid in in_use,
    }


def list_models() -> list[dict]:
    matched, local = _scan()
    in_use = _in_use()
    out = [_catalog_entry(m, matched, in_use) for m in catalog.CATALOG.values()]
    seen = {str(p) for p, _, _ in local}
    entries = [_local_entry(p, k, e, in_use) for p, k, e in local]
    # local models referenced by the config but outside the model folders
    cfg = load_config()
    for mid in [cfg.get("stt_model") or "", *(cfg.get("engine_models") or {}).values()]:
        if mid.startswith(LOCAL_PREFIX):
            p = Path(mid[len(LOCAL_PREFIX):])
            if str(p) not in seen and p.is_dir():
                kind = detect(p)
                if kind is not None:
                    seen.add(str(p))
                    entries.append(_local_entry(p, kind[0], kind[1], in_use))
    return out + entries


def model_info(mid: str) -> dict:
    for m in list_models():
        if m["id"] == mid:
            return m
    raise ModelNotFound(f"model {mid} not found")


def startup() -> None:
    load_config()
    refresh_meta_async()


def shutdown() -> None:
    for mid in list(_downloads):
        if _downloading(mid):
            try:
                cancel_download(mid)
            except Exception:  # noqa: BLE001
                log.exception("cancel download %s", mid)
