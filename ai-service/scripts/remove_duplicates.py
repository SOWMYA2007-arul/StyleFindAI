"""Find duplicate images with perceptual hashes (pHash).

A cryptographic hash (SHA-256) only catches byte-identical files. A perceptual hash
summarises how an image *looks* (low-frequency DCT structure), so re-saved / slightly
resized copies of the same photo get hashes that differ by only a few bits.

Used two ways:
  * imported by prepare_dataset.py while building the catalogue;
  * run directly to audit an existing catalogue:
        python scripts/remove_duplicates.py               (report only)
        python scripts/remove_duplicates.py --apply       (drop duplicate ROWS from catalogue.csv;
                                                           image files are never deleted)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import imagehash
from PIL import Image

from _common import CATALOGUE_CSV, PRODUCTS_DIR

PHASH_THRESHOLD = 3  # max differing bits (of 64) to call two images duplicates


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def image_phash(path: Path) -> imagehash.ImageHash:
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"))


def colour_thumb(path: Path):
    """32x32 colour thumbnail: confirms a pHash match really is the same picture, because
    pHash ignores colour (a red and a blue kurta with the same cut can share a pHash)."""
    import numpy as np

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)


THUMB_MAX_DIFF = 4.0  # mean absolute difference (0-255) between 32x32 colour thumbnails


def find_duplicates(items: list[tuple[str, Path]], threshold: int = PHASH_THRESHOLD) -> dict[str, str]:
    """items = [(product_id, image_path)] in priority order.
    Returns {duplicate_id: kept_id}. The first occurrence is kept."""
    seen_sha: dict[str, str] = {}
    kept: list[tuple[str, imagehash.ImageHash]] = []
    thumbs: list = []
    buckets: dict[str, list[int]] = {}  # prefix bucketing keeps this fast for thousands of images
    dups: dict[str, str] = {}
    for pid, path in items:
        sha = file_sha256(path)
        if sha in seen_sha:
            dups[pid] = seen_sha[sha]
            continue
        ph = image_phash(path)
        hexs = str(ph)
        match = None
        # compare against images sharing any 4-hex-char block (pigeonhole: <=3 bit diffs
        # leave at least one of four 16-bit blocks identical)
        cand = set()
        for b in range(4):
            cand.update(buckets.get(f"{b}:{hexs[b*4:(b+1)*4]}", []))
        for idx in cand:
            kid, kph = kept[idx]
            if ph - kph <= threshold:
                if float(abs(colour_thumb(path) - thumbs[idx]).mean()) <= THUMB_MAX_DIFF:
                    match = kid
                    break
        if match:
            dups[pid] = match
            continue
        seen_sha[sha] = pid
        kept.append((pid, ph))
        thumbs.append(colour_thumb(path))
        for b in range(4):
            buckets.setdefault(f"{b}:{hexs[b*4:(b+1)*4]}", []).append(len(kept) - 1)
    return dups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default=str(CATALOGUE_CSV))
    ap.add_argument("--apply", action="store_true", help="rewrite catalogue.csv without duplicate rows")
    args = ap.parse_args()
    with open(args.catalogue, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    items = [(r["product_id"], PRODUCTS_DIR / r["image_path"]) for r in rows if (PRODUCTS_DIR / r["image_path"]).exists()]
    dups = find_duplicates(items)
    print(f"Checked {len(items)} images: {len(dups)} duplicates")
    for d, k in list(dups.items())[:50]:
        print(f"  {d} duplicates {k}")
    if args.apply and dups:
        keep = [r for r in rows if r["product_id"] not in dups]
        with open(args.catalogue, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(keep)
        print(f"Removed {len(rows) - len(keep)} rows from {args.catalogue} (image files untouched)")


if __name__ == "__main__":
    main()
