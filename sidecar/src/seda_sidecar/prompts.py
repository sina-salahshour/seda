"""Bundled Persian read-aloud prompts (data/prompts_fa.txt, one sentence per line)."""
from __future__ import annotations

import random
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def all_prompts() -> tuple[str, ...]:
    path = config.BUNDLED_DATA / "prompts_fa.txt"
    seen: dict[str, None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            seen.setdefault(line, None)
    return tuple(seen)


def sample(n: int = 10) -> list[str]:
    pool = all_prompts()
    n = max(0, min(int(n), len(pool)))
    return random.sample(pool, n)
