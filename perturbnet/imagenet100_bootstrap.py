from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from perturbnet.constants import IMAGENET100_REPO_ID, IMAGENET100_SPLIT


def _repo_cache_dir(repo_id: str) -> Path:
    slug = repo_id.replace("/", "___")
    return Path.home() / ".cache" / "huggingface" / "datasets" / slug


def has_local_imagenet100_cache(repo_id: str = IMAGENET100_REPO_ID) -> bool:
    """True when processed Arrow shards exist under the HF datasets cache."""
    base = _repo_cache_dir(repo_id)
    if not base.is_dir():
        return False
    return any(base.rglob("*.arrow"))


def _resolve_offline(offline: bool | None, repo_id: str) -> bool:
    if offline is not None:
        return offline
    raw = os.getenv("PERTURB_IMAGENET100_OFFLINE", "auto").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return has_local_imagenet100_cache(repo_id=repo_id)


def load_imagenet100(
    repo_id: str = IMAGENET100_REPO_ID,
    split: str = IMAGENET100_SPLIT,
    *,
    offline: bool | None = None,
) -> Any:
    """Open the full ImageNet-100 split with random access.

    Uses the non-streaming `datasets` path: parquet/arrow shards are cached
    locally by Hugging Face and memory-mapped, so `dataset[i]` is cheap.

    When local Arrow cache exists (or PERTURB_IMAGENET100_OFFLINE=1), loads
    offline and does not contact huggingface.co.
    """
    try:
        from datasets import DownloadMode, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "`datasets` is required for ImageNet-100 challenges. "
            "Run `python -m pip install -r requirements.txt` first."
        ) from exc

    use_offline = _resolve_offline(offline, repo_id=repo_id)
    if use_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        source_note = "local Hugging Face datasets cache (offline)"
    else:
        source_note = "Hugging Face Hub (may download on first run)"

    split_candidates = [split]
    for candidate in ("train", "validation", "val", "test"):
        if candidate not in split_candidates:
            split_candidates.append(candidate)

    errors: list[str] = []
    for split_name in split_candidates:
        try:
            print(f"Loading ImageNet-100 repo={repo_id} split={split_name} from {source_note}")
            return load_dataset(
                repo_id,
                split=split_name,
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
            )
        except Exception as exc:
            errors.append(f"{split_name}: {exc}")

    raise RuntimeError("Unable to load ImageNet-100 dataset: " + " | ".join(errors))


def imagenet100_dataset_version(dataset: Any, repo_id: str, split: str) -> str:
    """Stable short identifier for a downloaded dataset snapshot.

    Intentionally based only on repo/split/row-count: it stays identical
    across restarts, `datasets` library upgrades, and cache rebuilds, so the
    validator's persisted traversal is never spuriously reset (which would
    allow duplicate image selections).
    """
    base = f"{repo_id}:{split}:{int(dataset.num_rows)}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
