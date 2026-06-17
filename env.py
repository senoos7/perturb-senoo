"""Load secrets from my_work/.env for scripts and local tooling."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
_loaded = False


def load_env() -> None:
    """Load my_work/.env into process environment (no-op if file missing)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not ENV_FILE.is_file():
        return
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def github_token() -> str:
    load_env()
    return os.getenv("GITHUB_TOKEN", "").strip() or os.getenv("GH_TOKEN", "").strip()
