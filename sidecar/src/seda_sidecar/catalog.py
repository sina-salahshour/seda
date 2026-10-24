"""Model catalog: every model Seda knows how to download and use.

``allow_patterns`` limits what is downloaded (and counted in size_mb), e.g.
openai/whisper-large-v3 also ships flax / .bin / fp32 copies (23 GB in total)
of which only model.safetensors + tokenizer files are needed.

Test hook: ``SEDA_EXTRA_CATALOG=/path/to.json`` (a JSON list of objects with the
CatalogModel fields) appends entries at import time. It exists only so the
download / progress / cancel path can be exercised with a tiny public repo; the
shipped catalog is the list below.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

_SAFETENSORS_ONLY = ("*.json", "*.txt", "*.model", "*.safetensors", "*.yaml", "*.py")


@dataclass(frozen=True)
class CatalogModel:
    id: str
    kind: str  # "tts" | "stt" | "aux"
    engine: str | None  # tts: the engine it runs on; aux: the engine that needs it; stt: None
    name: str
    description: str
    repo: str
    license: str
    commercial: bool
    gated: bool = False
    allow_patterns: tuple[str, ...] | None = None
    revision: str = "main"
    # extra kwargs for WhisperForConditionalGeneration.generate (stt only)
    generate_kwargs: dict = field(default_factory=dict)
    # env var that overrides where this model is loaded from (backward compatibility)
    env: str | None = None

    @property
    def repo_name(self) -> str:
        return self.repo.rsplit("/", 1)[-1]


_CATALOG: list[CatalogModel] = [
    CatalogModel(
        id="omnivoice",
        kind="tts",
        engine="omnivoice",
        name="OmniVoice",
        description="مدل اصلی شبیه‌سازی صدا، با پشتیبانی از آموزش LoRA (GPU). "
        "Main voice-cloning TTS, LoRA fine-tuning; includes its Higgs audio tokenizer.",
        repo="k2-fsa/OmniVoice",
        license="Apache-2.0 (audio_tokenizer/: Boson Higgs Audio 2 Community License)",
        commercial=True,
        env="SEDA_OMNIVOICE_MODEL",
    ),
    CatalogModel(
        id="higgs_tokenizer",
        kind="aux",
        engine="omnivoice",
        name="Higgs Audio v2 tokenizer",
        description="توکنایزر صوتی OmniVoice؛ فقط وقتی لازم است که پوشهٔ OmniVoice شامل audio_tokenizer نباشد. "
        "Audio tokenizer for OmniVoice training; only used when the OmniVoice folder has no audio_tokenizer/.",
        repo="eustlb/higgs-audio-v2-tokenizer",
        license="Boson Higgs Audio 2 Community License (Llama 3 based)",
        commercial=True,
    ),
    CatalogModel(
        id="pocket_v2",
        kind="tts",
        engine="pocket_v2",
        name="Pocket TTS Farsi v2",
        description="TTS سریع روی CPU؛ فقط استفادهٔ غیرتجاری. Fast CPU TTS, non-commercial only.",
        repo="mehdi-hf/pocket-tts-farsi-v2",
        license="CC-BY-NC-4.0",
        commercial=False,
        env="SEDA_POCKET_MODEL",
    ),
    CatalogModel(
        id="homo_ge2pe",
        kind="aux",
        engine="pocket_v2",
        name="Homo-GE2PE (Persian G2P)",
        description="تبدیل نویسه به واج برای Pocket TTS. Grapheme-to-phoneme model needed by pocket_v2.",
        repo="mehdi-hf/Homo-GE2PE-Persian-HF",
        license="MIT (base byt5-small: Apache-2.0)",
        commercial=True,
        env="SEDA_G2P_MODEL",
    ),
    CatalogModel(
        id="whisper_large_v3_turbo",
        kind="stt",
        engine=None,
        name="Whisper large-v3-turbo",
        description="تشخیص گفتار پیش‌فرض، سریع. Default speech recognition, fast (~1.6 GB).",
        repo="openai/whisper-large-v3-turbo",
        license="MIT",
        commercial=True,
        allow_patterns=_SAFETENSORS_ONLY,
    ),
    CatalogModel(
        id="whisper_large_v3",
        kind="stt",
        engine=None,
        name="Whisper large-v3",
        description="Whisper کامل، کندتر. Full Whisper large-v3, slower than turbo (~3 GB).",
        repo="openai/whisper-large-v3",
        license="Apache-2.0",
        commercial=True,
        allow_patterns=("*.json", "*.txt", "model.safetensors"),
    ),
    CatalogModel(
        id="buzzasr_persian",
        kind="stt",
        engine=None,
        name="BuzzASR Persian",
        description="Whisper large-v3 تنظیم‌شده برای فارسی، کمترین خطا. "
        "Whisper large-v3 fine-tuned for Persian, best Persian WER/CER (~3 GB).",
        repo="BuzzASR/persian",
        license="MIT",
        commercial=True,
        allow_patterns=_SAFETENSORS_ONLY,
        generate_kwargs={"no_repeat_ngram_size": 3, "repetition_penalty": 1.2},
    ),
    CatalogModel(
        id="parsvoice_xtts",
        kind="tts",
        engine="xtts",
        name="ParsVoice XTTS",
        description="XTTS-v2 تنظیم‌شده برای فارسی؛ غیرتجاری (CPML)، نیازمند توکن Hugging Face و پذیرش شرایط. "
        "XTTS-v2 fine-tuned on Persian (ParsVoice). Non-commercial (CPML); gated: accept the terms "
        "on huggingface.co and set a token.",
        repo="MohammadJRanjbar/ParsVoice-XTTS",
        license="CPML (Coqui Public Model License, non-commercial; the card says MIT, the XTTS-v2 base is CPML)",
        commercial=False,
        gated=True,
        allow_patterns=("*.json", "*.pth"),
    ),
]


def _extra() -> list[CatalogModel]:
    path = os.environ.get("SEDA_EXTRA_CATALOG")
    if not path:
        return []
    out = []
    for d in json.loads(open(path, encoding="utf-8").read()):
        if d.get("allow_patterns") is not None:
            d["allow_patterns"] = tuple(d["allow_patterns"])
        out.append(CatalogModel(**d))
    return out


CATALOG: dict[str, CatalogModel] = {m.id: m for m in _CATALOG + _extra()}
DEFAULT_STT = "whisper_large_v3_turbo"


def get(mid: str) -> CatalogModel | None:
    return CATALOG.get(mid)
