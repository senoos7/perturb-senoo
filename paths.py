"""Shared paths for my_work scripts and tools."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"

IMAGENET100_SAMPLES_DIR = DATA / "imagenet100_samples"
IMAGENET100_VAL_SAMPLES_DIR = DATA / "imagenet100_val_samples"

IMAGENET100_SHAPES_CACHE = DATA / "imagenet100_shapes.json"
IMAGENET100_LABELS_CACHE = DATA / "imagenet100_true_labels.json"
IMAGENET100_VAL_SHAPES_CACHE = DATA / "imagenet100_val_shapes.json"
IMAGENET100_VAL_LABELS_CACHE = DATA / "imagenet100_val_true_labels.json"
