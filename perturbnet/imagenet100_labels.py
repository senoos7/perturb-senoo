from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from perturbnet.constants import IMAGENET100_REPO_ID, IMAGENET100_SPLIT, MODEL_NAME

DEFAULT_LABELS_DIR = Path("data/imagenet100")
MANIFEST_FILENAME = "labels.jsonl"
META_FILENAME = "meta.json"


def labels_output_dir(root: Path | None = None) -> Path:
    base = root if root is not None else Path(__file__).resolve().parents[1]
    return base / DEFAULT_LABELS_DIR


def manifest_path(root: Path | None = None) -> Path:
    return labels_output_dir(root) / MANIFEST_FILENAME


def meta_path(root: Path | None = None) -> Path:
    return labels_output_dir(root) / META_FILENAME


def load_label_manifest(path: Path | None = None) -> dict[int, dict[str, Any]]:
    """Load cached label rows keyed by dataset row index."""
    target = path if path is not None else manifest_path()
    if not target.is_file():
        return {}
    records: dict[int, dict[str, Any]] = {}
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records[int(row["row"])] = row
    return records


def iter_label_manifest(path: Path | None = None) -> Iterator[dict[str, Any]]:
    target = path if path is not None else manifest_path()
    if not target.is_file():
        return iter(())
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_label_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def default_meta(*, dataset_rows: int, dataset_version: str) -> dict[str, Any]:
    return {
        "repo_id": IMAGENET100_REPO_ID,
        "split": IMAGENET100_SPLIT,
        "model_name": MODEL_NAME,
        "label_source": "efficientnet_v2_l_prediction",
        "dataset_rows": int(dataset_rows),
        "dataset_version": dataset_version,
        "manifest": MANIFEST_FILENAME,
        "notes": (
            "true_label matches validator challenges: EfficientNetV2-L prediction "
            "on JPEG(q=95) round-trip, normalized with normalize_prediction_label()."
        ),
    }
