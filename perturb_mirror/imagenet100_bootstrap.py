"""Mirror of perturbnet/imagenet100_bootstrap.py."""

from __future__ import annotations

import hashlib
from typing import Any

from perturb_mirror.constants import IMAGENET100_REPO_ID, IMAGENET100_SPLIT


def load_imagenet100(repo_id: str = IMAGENET100_REPO_ID, split: str = IMAGENET100_SPLIT) -> Any:
    """Download (once) and open the full ImageNet-100 split with random access."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "`datasets` is required for ImageNet-100 challenges. "
            "Run `python -m pip install -r requirements.txt` first."
        ) from exc

    split_candidates = [split]
    for candidate in ("train", "validation", "val", "test"):
        if candidate not in split_candidates:
            split_candidates.append(candidate)

    errors: list[str] = []
    for split_name in split_candidates:
        try:
            print(
                f"Loading ImageNet-100 repo={repo_id} split={split_name} "
                "(first run downloads the full split to the local Hugging Face cache)"
            )
            return load_dataset(repo_id, split=split_name)
        except Exception as exc:
            errors.append(f"{split_name}: {exc}")

    raise RuntimeError("Unable to load ImageNet-100 dataset from Hugging Face: " + " | ".join(errors))


def imagenet100_dataset_version(dataset: Any, repo_id: str, split: str) -> str:
    """Stable short identifier for a downloaded dataset snapshot."""
    base = f"{repo_id}:{split}:{int(dataset.num_rows)}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
