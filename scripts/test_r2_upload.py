#!/usr/bin/env python3
"""Verify Cloudflare R2 credentials and upload a test JPEG."""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from perturbnet.r2_storage import ChallengeR2Uploader


def _load_env_file(path: str) -> None:
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            os.environ[key] = value.strip().strip('"').strip("'")


def main() -> int:
    default_env = ROOT / "scripts" / "miner.env"
    parser = argparse.ArgumentParser(description="Test Cloudflare R2 challenge upload")
    parser.add_argument(
        "--env-file",
        default=str(default_env) if default_env.is_file() else "",
        help="Env file with PERTURB_R2_* vars (default: scripts/miner.env)",
    )
    args = parser.parse_args()
    if args.env_file:
        _load_env_file(args.env_file)

    uploader = ChallengeR2Uploader.from_env()
    image = Image.new("RGB", (64, 64), color=(30, 120, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    key = uploader.upload_jpeg(
        "test_r2_upload.jpg",
        buffer.getvalue(),
        metadata={"task_id": "test", "prompt": "test", "true_label": "test"},
    )
    print(f"upload_ok bucket={uploader.bucket} key={key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
