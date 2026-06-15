#!/usr/bin/env python3
"""Download Pexels search results for a class prompt (training dataset collection).

Downloads a single Pexels src variant only (default: medium), matching production
validator challenges. Photos without that variant are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perturbnet import constants as C


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
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value


def _resolve_image_url(src: dict[str, Any], image_variant: str) -> str:
    """Return src[image_variant] only — no fallback to large/original."""
    url = src.get(image_variant)
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"Pexels photo src is missing {image_variant!r} URL")
    return url.strip()


def _guess_extension(url: str, content_type: str | None) -> str:
    path = urlparse(url).path.lower()
    for ext in (".jpeg", ".jpg", ".png", ".webp"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    if content_type:
        lowered = content_type.lower()
        if "png" in lowered:
            return ".png"
        if "webp" in lowered:
            return ".webp"
    return ".jpg"


def _load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            photo_id = int(row["photo_id"])
            records[photo_id] = row
    return records


def _append_manifest(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_class_images(
    *,
    prompt: str,
    output_dir: Path,
    api_key: str,
    endpoint: str,
    image_variant: str,
    per_page: int,
    max_images: int | None,
    start_page: int,
    request_delay_seconds: float,
    download_delay_seconds: float,
    timeout_seconds: float,
) -> dict[str, int]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    meta_path = output_dir / "meta.json"

    manifest = _load_manifest(manifest_path)
    downloaded = sum(1 for row in manifest.values() if row.get("status") == "downloaded")
    skipped = 0
    failed = 0
    search_calls = 0
    page = max(1, start_page)
    total_results: int | None = None

    per_page = max(1, min(80, per_page))
    session = requests.Session()
    search_headers = {"Authorization": api_key}

    while True:
        if max_images is not None and downloaded >= max_images:
            break

        params = {"query": prompt, "page": page, "per_page": per_page}
        response = session.get(endpoint, params=params, headers=search_headers, timeout=timeout_seconds)
        response.raise_for_status()
        search_calls += 1
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Pexels search page {page} returned non-object JSON")

        if total_results is None:
            total_results = int(data.get("total_results") or 0)
            meta_path.write_text(
                json.dumps(
                    {
                        "prompt": prompt,
                        "endpoint": endpoint,
                        "image_variant": image_variant,
                        "per_page": per_page,
                        "total_results": total_results,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"prompt={prompt!r} total_results={total_results} variant={image_variant}")

        photos = data.get("photos")
        if not isinstance(photos, list) or not photos:
            print(f"page={page}: no photos, stopping")
            break

        for photo in photos:
            if max_images is not None and downloaded >= max_images:
                break
            if not isinstance(photo, dict):
                continue

            photo_id = int(photo.get("id") or 0)
            if photo_id <= 0:
                continue
            if photo_id in manifest and manifest[photo_id].get("status") == "downloaded":
                skipped += 1
                continue

            src = photo.get("src", {})
            if not isinstance(src, dict):
                src = {}

            try:
                image_url = _resolve_image_url(src, image_variant)
            except ValueError as exc:
                failed += 1
                row = {
                    "photo_id": photo_id,
                    "prompt": prompt,
                    "status": "failed",
                    "error": str(exc),
                    "page": page,
                }
                manifest[photo_id] = row
                _append_manifest(manifest_path, row)
                continue

            try:
                image_response = session.get(image_url, timeout=timeout_seconds)
                image_response.raise_for_status()
                image_bytes = image_response.content
                if not image_bytes:
                    raise ValueError("downloaded image is empty")
            except Exception as exc:
                failed += 1
                row = {
                    "photo_id": photo_id,
                    "prompt": prompt,
                    "status": "failed",
                    "error": str(exc),
                    "image_url": image_url,
                    "page": page,
                }
                manifest[photo_id] = row
                _append_manifest(manifest_path, row)
                if download_delay_seconds > 0:
                    time.sleep(download_delay_seconds)
                continue

            ext = _guess_extension(image_url, image_response.headers.get("Content-Type"))
            filename = f"pexels-{photo_id}{ext}"
            out_path = images_dir / filename
            out_path.write_bytes(image_bytes)

            row = {
                "photo_id": photo_id,
                "prompt": prompt,
                "status": "downloaded",
                "page": page,
                "image_variant": image_variant,
                "image_url": image_url,
                "local_path": str(out_path.relative_to(ROOT)),
                "filename": filename,
                "bytes": len(image_bytes),
                "width": photo.get("width"),
                "height": photo.get("height"),
                "photographer": photo.get("photographer"),
                "photographer_url": photo.get("photographer_url"),
                "pexels_url": photo.get("url"),
            }
            manifest[photo_id] = row
            _append_manifest(manifest_path, row)
            downloaded += 1

            if downloaded % 25 == 0:
                print(f"downloaded={downloaded} page={page} latest={filename}")

            if download_delay_seconds > 0:
                time.sleep(download_delay_seconds)

        max_page = max(1, (total_results + per_page - 1) // per_page) if total_results else page
        if page >= max_page:
            print(f"reached last page ({page}/{max_page})")
            break
        page += 1
        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "search_calls": search_calls,
        "pages_scanned": page - start_page + 1,
    }


def main() -> int:
    default_env = ROOT / "scripts" / "validator.env"
    parser = argparse.ArgumentParser(description="Fetch Pexels images for one class prompt")
    parser.add_argument("--prompt", default="dog", help="Search query (class prompt)")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: data/pexels/<prompt-slug>)",
    )
    parser.add_argument(
        "--env-file",
        default=str(default_env) if default_env.is_file() else "",
        help="Env file with PERTURB_PEXELS_API_KEY (default: scripts/validator.env)",
    )
    parser.add_argument("--endpoint", default="", help=f"Override {C.IMAGE_ENDPOINT}")
    parser.add_argument(
        "--image-variant",
        default="medium",
        help="Pexels src key to download (default: medium only, no fallback)",
    )
    parser.add_argument("--per-page", type=int, default=0, help=f"Override per_page (max 80, default {C.PEXELS_PER_PAGE})")
    parser.add_argument("--max-images", type=int, default=0, help="Stop after N new downloads (0 = all)")
    parser.add_argument("--start-page", type=int, default=1, help="Resume from this search page")
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.35,
        help="Seconds between Pexels search API calls",
    )
    parser.add_argument(
        "--download-delay",
        type=float,
        default=0.05,
        help="Seconds between image downloads",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds")
    args = parser.parse_args()

    if args.env_file:
        _load_env_file(args.env_file)

    api_key = (os.getenv("PERTURB_PEXELS_API_KEY") or os.getenv("PEXELS_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit(
            "Missing Pexels API key. Set PERTURB_PEXELS_API_KEY or pass --env-file scripts/validator.env"
        )

    prompt = args.prompt.strip()
    if not prompt:
        raise SystemExit("Empty --prompt")

    slug = prompt.lower().replace(" ", "_")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "data" / "pexels" / slug
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    endpoint = args.endpoint.strip() or os.getenv(
        "PERTURB_IMAGE_ENDPOINT", "https://api.pexels.com/v1/search"
    )
    image_variant = args.image_variant.strip().lower() or "medium"
    if args.per_page > 0:
        per_page = args.per_page
    else:
        raw_per_page = os.getenv("PERTURB_PEXELS_PER_PAGE", str(C.PEXELS_PER_PAGE))
        try:
            per_page = int(raw_per_page)
        except ValueError:
            per_page = C.PEXELS_PER_PAGE
    max_images = args.max_images if args.max_images > 0 else None

    print(f"output_dir={output_dir}")
    stats = fetch_class_images(
        prompt=prompt,
        output_dir=output_dir,
        api_key=api_key,
        endpoint=endpoint,
        image_variant=image_variant,
        per_page=per_page,
        max_images=max_images,
        start_page=max(1, args.start_page),
        request_delay_seconds=max(0.0, args.request_delay),
        download_delay_seconds=max(0.0, args.download_delay),
        timeout_seconds=max(1.0, args.timeout),
    )
    print(
        "done: "
        + " ".join(f"{key}={value}" for key, value in stats.items())
        + f" manifest={output_dir / 'manifest.jsonl'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
