"""Engine interface. Every TTS engine returns float32 mono audio at 24 kHz."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..store import VoiceProfile


class Engine(ABC):
    id: str
    name: str
    uses_gpu: bool

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    @abstractmethod
    def load(self) -> None:
        """Load weights. Called lazily by the manager, under the GPU lock."""

    @abstractmethod
    def unload(self) -> None:
        """Free all weights (and call torch.cuda.empty_cache() if on GPU)."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        profile: VoiceProfile,
        run_dir: Path | None,
        speed: float = 1.0,
    ) -> np.ndarray:
        """Synthesize raw (un-normalised) Persian text in the profile's voice.

        run_dir: profiles/<pid>/runs/<rid> of a finished run, or None for zero-shot.
          omnivoice uses run_dir/"adapter" (PEFT LoRA) when present;
          pocket_v2 uses run_dir/"pocket" (chosen reference clip + cached voice state)
          when present, otherwise picks a reference from the profile on the fly.
        Returns np.float32 mono @ 24 kHz in [-1, 1].
        """
