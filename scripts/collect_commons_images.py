"""Collect additional real product photos from Wikimedia Commons."""

from __future__ import annotations

import argparse
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
)

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "RetailVision/0.3 (https://github.com/; educational dataset)"
QUERIES = {
    1: ["Lay's potato chips", "Lays Classic chips bag"],
    4: ["Pepsi bottle", "Pepsi-Cola 500ml"],
    5: ["Sprite soda bottle", "Sprite drink can"],
    7: ["Kurkure", "Kurkure Masala Munch"],
}


def _search(query: str, limit: int = 12) -> list[dict]:
    response = requests.get(
        COMMONS_API,
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "iiurlwidth": 1024,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    pages = (response.json().get("query") or {}).get("pages") or {}
    return list(pages.values())


def collect(per_class: int) -> None:
    ensure_dataset_dirs()
    products = {int(item["class_id"]): item for item in load_registry()}
    manifest = load_manifest()
    existing_urls = {entry.get("source_url") for entry in manifest}

    for class_id, queries in QUERIES.items():
        product = products[class_id]
        have = sum(1 for entry in manifest if int(entry["class_id"]) == class_id)
        print(f"Commons class {class_id} {product['name']} have={have}", flush=True)
        for query in queries:
            if have >= per_class:
                break
            try:
                pages = _search(query)
            except requests.RequestException as exc:
                print(f"  search failed: {exc}", flush=True)
                continue
            time.sleep(0.2)
            for page in pages:
                if have >= per_class:
                    break
                info = (page.get("imageinfo") or [{}])[0]
                mime = str(info.get("mime") or "")
                url = info.get("thumburl") or info.get("url")
                page_id = str(page.get("pageid") or page.get("title") or "")
                if not url or not mime.startswith("image/") or url in existing_urls:
                    continue
                try:
                    payload = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
                    payload.raise_for_status()
                except requests.RequestException:
                    continue
                temp = RAW_DIR / "images" / "_tmp_download.jpg"
                temp.write_bytes(payload.content)
                image = load_image(temp)
                temp.unlink(missing_ok=True)
                if image is None or min(image.shape[:2]) < 180:
                    continue
                saved_for = sum(
                    1
                    for entry in manifest
                    if int(entry["class_id"]) == class_id and str(entry.get("session_id")) == f"{class_id}_cm{page_id}"
                )
                filename = f"c{class_id}_cm{page_id}_{saved_for:02d}.jpg"
                dest = RAW_DIR / "images" / filename
                save_image(dest, image)
                height, width = image.shape[:2]
                manifest.append(
                    {
                        "file": filename,
                        "class_id": class_id,
                        "class_name": product["name"],
                        "session_id": f"{class_id}_cm{page_id}",
                        "barcode": f"cm{page_id}",
                        "source_url": url,
                        "source": "wikimedia_commons",
                        "product_name": page.get("title"),
                        "width": int(width),
                        "height": int(height),
                    }
                )
                existing_urls.add(url)
                save_manifest(manifest)
                have += 1
                print(f"  saved {filename}", flush=True)
                time.sleep(0.15)
        print(f"  total class {class_id}: {have}", flush=True)
    print(f"Raw images: {len(manifest)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Wikimedia Commons product photos.")
    parser.add_argument("--per-class", type=int, default=8)
    args = parser.parse_args()
    collect(args.per_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
