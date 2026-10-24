"""Training worker (run as a subprocess by ``training``).

    python -m seda_sidecar.train_worker <run_dir>

Reads ``<run_dir>/job.json`` (written by training.start_run):
    {"profile_id", "engine", "samples": [{"id", "wav", "text", "duration"}],
     "params": {...}, "from_run_dir": str|null, "model_paths": {role: path|repo}}

and calls ``train(job, run_dir, emit)`` of the job's engine (see engines.REGISTRY):
  omnivoice  LoRA fine-tuning (engines/omnivoice_train.py) -> run_dir/adapter
  pocket_v2  best <= 5 s reference clip -> run_dir/pocket/{ref.wav, voice.safetensors}
  xtts       speaker conditioning latents -> run_dir/xtts/latents.pt

Progress is reported on stdout as lines ``@@PROGRESS {json}`` which the parent
(training.py) turns into run.json updates; everything else is plain log text.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROGRESS_PREFIX = "@@PROGRESS "


def emit(**kw) -> None:
    print(PROGRESS_PREFIX + json.dumps(kw, ensure_ascii=False), flush=True)


_T0 = time.time()


def logp(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.1f}s]", *a, flush=True)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m seda_sidecar.train_worker <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1]).resolve()
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from .engines import REGISTRY

    eid = job.get("engine") or "omnivoice"
    if eid not in REGISTRY:
        print(f"unknown engine {eid!r}", flush=True)
        return 2
    logp(f"seda train_worker: run {run_dir.name}, engine {eid}, {len(job['samples'])} samples, "
         f"models {json.dumps(job.get('model_paths') or {}, ensure_ascii=False)}, "
         f"params {json.dumps(job['params'])}")
    emit(stage="start", progress=0.0, message="starting")
    engine = REGISTRY[eid].load_class()()
    engine.train(job, run_dir, emit)
    emit(stage="done", progress=1.0, message="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
