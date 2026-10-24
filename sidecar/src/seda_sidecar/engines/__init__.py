"""Engine registry: the single source of truth for TTS engine ids and their metadata.

Everything that needs to know which engines exist (store.ENGINES, server
validation, manager, /health, the training worker and the training job
manager) derives it from ``REGISTRY``. Engine classes are imported lazily so
importing this package stays cheap and free of torch / circular imports.

Per engine:
  cls          "module:Class" (relative to this package), imported on demand
  name         display name
  uses_gpu     inference runs on the GPU (manager keeps one GPU engine loaded)
  train_gpu    the training worker needs the GPU (sidecar + Ollama models are
               unloaded first and GPU inference answers 409 meanwhile)
  trainable    step-based fine-tuning (``steps`` / ``from_run`` apply). Engines with
               trainable=False still implement train() (POST /train "prepares the
               voice": reference clip / latents) and ignore steps / from_run.
  license      license of the engine's weights (short text)
  commercial   whether the weights allow commercial use
  models       required model catalog ids; models[0] is the engine's main model
               (overridable per engine with models.json "engine_models")
  default_params() / default_steps(total_seconds)   training defaults
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------- training defaults
def _omnivoice_params() -> dict:
    return {
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.05,
        "batch_tokens": 3072,
        "grad_accum": 2,
        "max_batch_size": 16,
        "max_sample_tokens": 1500,
        "min_sample_tokens": 30,
        "gradient_checkpointing": True,
        "mixed_precision": "bf16",
        "attn_implementation": "sdpa",
        "seed": 42,
    }


def _omnivoice_steps(total_seconds: float) -> int:
    """clamp(round(150 + 12 * minutes), 200, 1000): roughly 40-60 passes over the data."""
    minutes = total_seconds / 60.0
    return int(min(1000, max(200, round(150 + 12 * minutes))))


def _pocket_params() -> dict:
    return {"max_reference_seconds": 5.0}


def _xtts_params() -> dict:
    return {"max_reference_seconds": 45.0, "max_clips": 12, "gpt_cond_chunk_len": 6}


def _one_step(total_seconds: float) -> int:
    return 1


@dataclass(frozen=True)
class EngineSpec:
    id: str
    name: str
    cls: str
    uses_gpu: bool
    train_gpu: bool
    trainable: bool
    license: str
    commercial: bool
    models: tuple[str, ...]
    default_params: Callable[[], dict] = field(default=dict)
    default_steps: Callable[[float], int] = field(default=_one_step)

    def load_class(self):
        mod, _, name = self.cls.partition(":")
        return getattr(importlib.import_module(mod, __name__), name)


REGISTRY: dict[str, EngineSpec] = {
    "omnivoice": EngineSpec(
        id="omnivoice",
        name="OmniVoice",
        cls=".omnivoice_engine:OmniVoiceEngine",
        uses_gpu=True,
        train_gpu=True,
        trainable=True,
        license="Apache-2.0 (audio tokenizer: Boson Higgs Audio 2 Community License)",
        commercial=True,
        models=("omnivoice",),
        default_params=_omnivoice_params,
        default_steps=_omnivoice_steps,
    ),
    "pocket_v2": EngineSpec(
        id="pocket_v2",
        name="Pocket TTS Farsi v2",
        cls=".pocket_engine:PocketV2Engine",
        uses_gpu=False,
        train_gpu=False,
        trainable=False,
        license="CC-BY-NC-4.0 (non-commercial)",
        commercial=False,
        models=("pocket_v2", "homo_ge2pe"),
        default_params=_pocket_params,
        default_steps=_one_step,
    ),
    "xtts": EngineSpec(
        id="xtts",
        name="ParsVoice XTTS",
        cls=".xtts_engine:XttsEngine",
        uses_gpu=True,
        train_gpu=False,
        trainable=False,
        license="CPML (Coqui Public Model License, non-commercial)",
        commercial=False,
        models=("parsvoice_xtts",),
        default_params=_xtts_params,
        default_steps=_one_step,
    ),
}

ENGINE_IDS: tuple[str, ...] = tuple(REGISTRY)
DEFAULT_ENGINE = "omnivoice"


def spec(eid: str) -> EngineSpec:
    try:
        return REGISTRY[eid]
    except KeyError:
        raise KeyError(f"unknown engine {eid!r}") from None


def default_params(eid: str) -> dict:
    return dict(spec(eid).default_params())


def default_steps(eid: str, total_seconds: float) -> int:
    return int(spec(eid).default_steps(total_seconds))
