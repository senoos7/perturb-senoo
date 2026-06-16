#!/usr/bin/env python3
"""Deprecated wrapper — use scripts/imagenet100_local_test.py instead."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "imagenet100_local_test.py"


def main() -> int:
    print(
        "local_e2e_test.py is deprecated (Pexels single-image path removed). "
        "Running imagenet100_local_test.py with --max-rows 1.",
        file=sys.stderr,
    )
    cmd = [sys.executable, str(TARGET), "--max-rows", "1", *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
