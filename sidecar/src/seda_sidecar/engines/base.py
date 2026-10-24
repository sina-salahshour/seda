"""Engine interface. Every TTS engine returns float32 mono audio at 24 kHz."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np

from ..store import VoiceProfile


class Engine(ABC):
    """Metadata (name, uses_gpu, license, ...) lives in ``engines.REGISTRY``."""

    id: str
    name: str
    uses_gpu: bool
    # {model id: resolved path or repo} of the weights currently loaded; set by
    # load() (see models.engine_paths), used for GET /models "in_use".
    loaded_models: dict[str, str] = {}

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

    # ------------------------------------------------------------ training
    def train(self, job: dict, run_dir: Path, emit: Callable[..., None]) -> None:
        """Produce this engine's artefact in run_dir. Runs inside the train_worker
        subprocess (never in the server process). ``job`` is run_dir/job.json:
        {"profile_id", "engine", "samples": [{"id", "wav", "text", "duration"}],
         "params": {...}, "from_run_dir": str|null, "model_paths": {role: path|repo}}.
        ``emit(stage=..., progress=0..1, message=..., step=..., total_steps=..., loss=...)``
        reports progress to the parent."""
        raise NotImplementedError(f"engine {self.id} cannot be trained")

    def artifact_ok(self, run_dir: Path) -> bool:
        """True if run_dir holds a usable artefact written by train()."""
        return False
