"""Validate the clean catalogue (catalogue.csv + dataset/products).

Checks: required columns, empty values, duplicate IDs, files exist, images decode, allowed
format, minimum dimensions, unwanted categories, minimum catalogue size.
Exit code 0 = all good, 1 = problems found (so it can be used in CI).

    python scripts/validate_images.py --min-count 1000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter

from PIL import Image

from _common import CATALOGUE_COLUMNS, CATALOGUE_CSV, CATEGORY_GROUPS, EXCLUDED_ARTICLE, PRODUCTS_DIR

REQUIRED_NON_EMPTY = ["product_id", "product_name", "category", "image_path", "source_dataset", "licence_information"]


def validate(catalogue=CATALOGUE_CSV, min_count: int = 1000) -> dict:
    problems: list[str] = []
    with open(catalogue, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in CATALOGUE_COLUMNS[:11] if c not in (reader.fieldnames or [])]
        rows = list(reader)
    if missing_cols:
        problems.append(f"missing columns: {missing_cols}")
    ids = Counter(r["product_id"] for r in rows)
    for pid, n in ids.items():
        if n > 1:
            problems.append(f"duplicate product_id {pid} x{n}")
    paths = Counter(r["image_path"] for r in rows)
    for p, n in paths.items():
        if n > 1:
            problems.append(f"image used by {n} products: {p}")
    allowed_articles = {a for v in CATEGORY_GROUPS.values() for a in v}
    stats = Counter()
    for r in rows:
        for c in REQUIRED_NON_EMPTY:
            if not (r.get(c) or "").strip():
                problems.append(f"{r.get('product_id')}: empty {c}")
        if r["category"] not in CATEGORY_GROUPS:
            problems.append(f"{r['product_id']}: unknown category {r['category']}")
        if r.get("subcategory") in EXCLUDED_ARTICLE or r.get("subcategory") not in allowed_articles:
            problems.append(f"{r['product_id']}: unwanted subcategory {r.get('subcategory')}")
        if ".." in r["image_path"] or r["image_path"].startswith(("/", "\\")) or ":" in r["image_path"]:
            problems.append(f"{r['product_id']}: unsafe image_path {r['image_path']}")
            continue
        p = PRODUCTS_DIR / r["image_path"]
        if not p.exists():
            problems.append(f"{r['product_id']}: image missing ({r['image_path']})"); continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                fmt, (w, h) = im.format, im.size
                im.convert("RGB").load()
            if fmt not in ("JPEG", "PNG", "WEBP"):
                problems.append(f"{r['product_id']}: format {fmt}")
            if min(w, h) < 60:
                problems.append(f"{r['product_id']}: too small {w}x{h}")
            stats["ok_images"] += 1
        except Exception as e:  # noqa: BLE001
            problems.append(f"{r['product_id']}: broken image ({type(e).__name__})")
    if len(rows) < min_count:
        problems.append(f"catalogue has {len(rows)} products, fewer than required {min_count}")
    return {"rows": len(rows), "ok_images": stats["ok_images"], "problems": problems, "passed": not problems}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=1000)
    args = ap.parse_args()
    res = validate(min_count=args.min_count)
    print(json.dumps({k: v for k, v in res.items() if k != "problems"}, indent=2))
    for p in res["problems"][:100]:
        print("PROBLEM:", p)
    print("PASSED" if res["passed"] else f"FAILED ({len(res['problems'])} problems)")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
