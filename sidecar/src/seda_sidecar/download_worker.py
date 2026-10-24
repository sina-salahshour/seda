"""Download one HF repo into the normal HF cache (run as a child of ``models``).

    python -m seda_sidecar.download_worker '{"repo", "revision", "allow_patterns"}'

The token comes from the HF_TOKEN env var. On failure the last stdout line is
``@@ERROR {"type", "message", "status", "gated"}`` and the exit code is 1.
"""
from __future__ import annotations

import json
import signal
import sys

ERROR_PREFIX = "@@ERROR "


def main() -> int:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    job = json.loads(sys.argv[1])
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=job["repo"],
            revision=job.get("revision") or "main",
            allow_patterns=job.get("allow_patterns"),
            max_workers=4,
        )
        print(f"downloaded {job['repo']} to {path}", flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        status = None
        resp = getattr(e, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
        names = {c.__name__ for c in type(e).__mro__}
        cause = e.__cause__ or e.__context__
        if cause is not None:
            names |= {c.__name__ for c in type(cause).__mro__}
        err = {
            "type": type(e).__name__,
            "message": str(e).splitlines()[0] if str(e) else type(e).__name__,
            "status": status,
            "gated": "GatedRepoError" in names,
        }
        print(f"{type(e).__name__}: {e}", flush=True)
        print(ERROR_PREFIX + json.dumps(err, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
