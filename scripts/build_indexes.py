#!/usr/bin/env python3
"""
Rebuild consolidated shape + true-label index caches from local .npy / .json files.

No Hugging Face required. Run after save_images + run_inference + save_decoded_npy.

Usage (from my_work/):
  python scripts/build_indexes.py --split train
  python scripts/build_indexes.py --split validation
  python scripts/build_indexes.py --split all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from local_index import (
    build_label_index_from_json,
    build_shape_index_from_npy,
    dataset_version,
    discover_npy_rows,
)
from paths import (
    IMAGENET100_LABELS_CACHE,
    IMAGENET100_SAMPLES_DIR,
    IMAGENET100_SHAPES_CACHE,
    IMAGENET100_VAL_LABELS_CACHE,
    IMAGENET100_VAL_SAMPLES_DIR,
    IMAGENET100_VAL_SHAPES_CACHE,
)
from perturb_mirror.constants import IMAGENET100_REPO_ID


SPLIT_CONFIG = {
    "train": {
        "hf_split": "train",
        "samples_dir": IMAGENET100_SAMPLES_DIR,
        "shape_cache": IMAGENET100_SHAPES_CACHE,
        "label_cache": IMAGENET100_LABELS_CACHE,
    },
    "validation": {
        "hf_split": "validation",
        "samples_dir": IMAGENET100_VAL_SAMPLES_DIR,
        "shape_cache": IMAGENET100_VAL_SHAPES_CACHE,
        "label_cache": IMAGENET100_VAL_LABELS_CACHE,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild shape/label index caches from local files")
    p.add_argument(
        "--split",
        choices=["train", "validation", "all"],
        default="all",
        help="Which prepared split to index (default: all)",
    )
    p.add_argument("--log-every", type=int, default=5000)
    return p.parse_args()


def rebuild_one(name: str, cfg: dict, log_every: int) -> None:
    samples_dir: Path = cfg["samples_dir"]
    if not samples_dir.is_dir():
        print(f"[{name}] skip — directory not found: {samples_dir}")
        return

    rows = discover_npy_rows(samples_dir)
    if not rows:
        print(f"[{name}] skip — no .npy files in {samples_dir}")
        return

    version = dataset_version(cfg["hf_split"], len(rows))
    print(f"\n[{name}] rows={len(rows)}  version={version}")
    print(f"  samples : {samples_dir}")
    print(f"  shapes  : {cfg['shape_cache']}")
    print(f"  labels  : {cfg['label_cache']}")

    build_shape_index_from_npy(
        samples_dir=samples_dir,
        rows=rows,
        cache_path=cfg["shape_cache"],
        version=version,
        log_every=log_every,
    )
    build_label_index_from_json(
        samples_dir=samples_dir,
        rows=rows,
        cache_path=cfg["label_cache"],
        version=version,
        log_every=log_every,
    )


def main() -> int:
    args = parse_args()
    print(f"repo    : {IMAGENET100_REPO_ID}")

    names = list(SPLIT_CONFIG) if args.split == "all" else [args.split]
    for name in names:
        rebuild_one(name, SPLIT_CONFIG[name], args.log_every)

    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
