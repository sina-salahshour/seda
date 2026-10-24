"""Process-wide configuration (env vars) and paths."""
from __future__ import annotations

import os
from pathlib import Path

HOST = os.environ.get("SEDA_HOST", "127.0.0.1")
PORT = int(os.environ.get("SEDA_PORT", "8765"))
DATA_DIR = Path(os.environ.get("SEDA_DATA", Path.home() / ".local/share/seda")).expanduser()
PROFILES_DIR = DATA_DIR / "profiles"
OLLAMA_URL = os.environ.get("SEDA_OLLAMA_URL", "http://127.0.0.1:11434")
SAMPLE_RATE = 24_000
PACKAGE_DIR = Path(__file__).resolve().parent
BUNDLED_DATA = PACKAGE_DIR / "data"

# Model ids (HF hub). Overridable for offline/local mirrors.
OMNIVOICE_MODEL = os.environ.get("SEDA_OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
POCKET_MODEL = os.environ.get("SEDA_POCKET_MODEL", "mehdi-hf/pocket-tts-farsi-v2")
G2P_MODEL = os.environ.get("SEDA_G2P_MODEL", "mehdi-hf/Homo-GE2PE-Persian-HF")
WHISPER_MODEL = os.environ.get("SEDA_WHISPER_MODEL", "openai/whisper-large-v3-turbo")
