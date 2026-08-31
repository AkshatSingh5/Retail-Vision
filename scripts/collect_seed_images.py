"""Collect real product photographs from Open Food Facts into dataset/raw.

Images are grouped by barcode (session) so later splits do not leak
near-duplicate shots of the same pack into train and test.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.dataset_common import (
    RAW_DIR,
    ensure_dataset_dirs,
    load_image,
    load_manifest,
    load_registry,
    save_image,
    save_manifest,
    write_data_yaml,
)

OFF_SEARCH = "https://world.openfoodfacts.org/cgi/search.pl"
OFF_PRODUCT = "https://world.openfoodfacts.org/api/v2/product/{code}.json"
USER_AGENT = "RetailVision/0.3 (dataset collection; educational)"
MIN_SIDE = 180
FILENAME_RE = re.compile(r"^c(\d+)_([A-Za-z0-9]+)_(\d+)\.jpg$")


def _safe(text: str) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def _matches(name: str, product: dict) -> bool:
    lowered = name.lower()
    includes = [token.lower() for token in product.get("name_include") or [] if token]
    excludes = [token.lower() for token in product.get("name_exclude") or [] if token]
    if includes and not any(token in lowered for token in includes):
        return False
    if any(token in lowered for token in excludes):
        return False
    return True


def _get_json(url: str, params: dict | None = None, attempts: int = 4) -> dict | None:
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            if response.status_code == 503:
                time.sleep(1.5 * attempt)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            time.sleep(1.0 * attempt)
    return None


def _search(term: str, page_size: int = 24) -> list[dict]:
    payload = _get_json(
        OFF_SEARCH,
        {
            "search_terms": term,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": page_size,
            "fields": "code,product_name,image_url,image_front_url,image_ingredients_url,image_packaging_url",
        },
    )
    if not payload:
        return []
    return list(payload.get("products") or [])


def _product_by_barcode(code: str) -> dict | None:
    payload = _get_json(OFF_PRODUCT.format(code=code))
    if not payload or int(payload.get("status") or 0) != 1:
        return None
    return payload.get("product")


def _candidate_urls(item: dict) -> list[str]:
    urls: list[str] = []
    selected = item.get("selected_images") or {}
    front = ((selected.get("front") or {}).get("display") or {})
    for lang_url in front.values():
        if lang_url and lang_url not in urls:
            urls.append(lang_url)
    for key in ("image_front_url", "image_url", "image_packaging_url"):
        url = item.get(key)
        if url and url not in urls:
            urls.append(url)
    return urls


def _download(url: str) -> bytes | None:
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None


def _sync_existing_manifest(manifest: list[dict], products: list[dict]) -> list[dict]:
    names = {int(item["class_id"]): str(item["name"]) for item in products}
    known = {entry["file"] for entry in manifest}
    images_dir = RAW_DIR / "images"
    if not images_dir.exists():
        return manifest
    for path in sorted(images_dir.glob("*.jpg")):
        if path.name in known:
            continue
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        class_id = int(match.group(1))
        barcode = match.group(2)
        image = load_image(path)
        height, width = (image.shape[:2] if image is not None else (0, 0))
        manifest.append(
            {
                "file": path.name,
                "class_id": class_id,
                "class_name": names.get(class_id, f"class {class_id}"),
                "session_id": f"{class_id}_{barcode}",
                "barcode": barcode,
                "source_url": None,
                "source": "openfoodfacts",
                "width": int(width),
                "height": int(height),
            }
        )
        known.add(path.name)
    return manifest


def _save_product_images(
    item: dict,
    product: dict,
    max_images_per_session: int,
    sessions: dict[str, int],
    manifest: list[dict],
    existing_keys: set[tuple],
) -> None:
    class_id = int(product["class_id"])
    barcode = str(item.get("code") or "").strip()
    name = str(item.get("product_name") or "")
    if not barcode:
        return
    saved_for_barcode = sessions.get(barcode, 0)
    if saved_for_barcode >= max_images_per_session:
        return
    for url in _candidate_urls(item):
        if saved_for_barcode >= max_images_per_session:
            break
        if (url, class_id) in existing_keys:
            continue
        payload = _download(url)
        time.sleep(0.2)
        if not payload:
            continue
        temp_path = RAW_DIR / "images" / "_tmp_download.jpg"
        temp_path.write_bytes(payload)
        image = load_image(temp_path)
        temp_path.unlink(missing_ok=True)
        if image is None:
            continue
        height, width = image.shape[:2]
        if min(height, width) < MIN_SIDE:
            continue
        stem = f"c{class_id}_{barcode}_{saved_for_barcode:02d}"
        image_name = f"{stem}.jpg"
        dest = RAW_DIR / "images" / image_name
        if dest.exists():
            saved_for_barcode += 1
            sessions[barcode] = saved_for_barcode
            continue
        save_image(dest, image)
        entry = {
            "file": image_name,
            "class_id": class_id,
            "class_name": product["name"],
            "session_id": f"{class_id}_{barcode}",
            "barcode": barcode,
            "source_url": url,
            "source": "openfoodfacts",
            "product_name": name,
            "width": int(width),
            "height": int(height),
        }
        manifest.append(entry)
        existing_keys.add((url, class_id))
        sessions[barcode] = saved_for_barcode + 1
        saved_for_barcode += 1
        save_manifest(manifest)
        print(f"  saved {image_name} ({_safe(name)[:60]})")


def collect(max_sessions: int, max_images_per_session: int) -> None:
    ensure_dataset_dirs()
    write_data_yaml()
    products = load_registry()
    manifest = _sync_existing_manifest(load_manifest(), products)
    existing_keys = {(entry.get("source_url"), entry.get("class_id")) for entry in manifest}

    for product in products:
        class_id = int(product["class_id"])
        sessions: dict[str, int] = {}
        for entry in manifest:
            if int(entry["class_id"]) == class_id and entry.get("barcode"):
                sessions[str(entry["barcode"])] = sessions.get(str(entry["barcode"]), 0) + 1
        print(f"Collecting class {class_id}: {_safe(product['name'])}")

        for code in product.get("barcodes") or []:
            if len(sessions) >= max_sessions and code not in sessions:
                continue
            item = _product_by_barcode(str(code))
            time.sleep(0.25)
            if not item:
                continue
            name = str(item.get("product_name") or "")
            if name and not _matches(name, product) and not any(
                token.lower() in name.lower() for token in (product.get("name_include") or [])
            ):
                # Barcode list is curated; keep unless clearly wrong.
                pass
            _save_product_images(
                item, product, max_images_per_session, sessions, manifest, existing_keys
            )

        for term in product.get("search_terms") or [product["name"]]:
            if len(sessions) >= max_sessions:
                break
            hits = _search(term)
            if not hits:
                print(f"  search unavailable for {_safe(term)!r}")
                continue
            for item in hits:
                name = str(item.get("product_name") or "")
                if not _matches(name, product):
                    continue
                barcode = str(item.get("code") or "").strip()
                if not barcode:
                    continue
                if barcode not in sessions and len(sessions) >= max_sessions:
                    continue
                _save_product_images(
                    item, product, max_images_per_session, sessions, manifest, existing_keys
                )
        print(f"  sessions={len(sessions)} images={sum(sessions.values())}")
    save_manifest(manifest)
    print(f"Raw images: {len(manifest)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect real retail product photos.")
    parser.add_argument("--max-sessions", type=int, default=8)
    parser.add_argument("--max-images-per-session", type=int, default=3)
    args = parser.parse_args()
    collect(args.max_sessions, args.max_images_per_session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
