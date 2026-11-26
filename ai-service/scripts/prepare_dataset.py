"""Build the clean StyleFind catalogue from the Kaggle 'Fashion Product Images' dataset.

Steps (matching the project brief):
  read CSV -> women's clothing only -> drop footwear / bags / watches / jewellery / personal care /
  innerwear / sleepwear / other categories -> drop duplicate IDs -> check image exists ->
  validate format + dimensions -> drop duplicate images (pHash) -> pick per-category quotas ->
  copy images into category folders (filename = original product id) -> write catalogue.csv

Usage (PowerShell, inside ai-service with the venv active):
    python scripts/prepare_dataset.py --raw dataset/raw/fashion-product-images-small
    python scripts/prepare_dataset.py --raw dataset/raw/synthetic-fixture --source SYNTHETIC_TEST_FIXTURE --licence "test-only"
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from _common import (CATALOGUE_COLUMNS, CATALOGUE_CSV, CATEGORY_GROUPS, DEFAULT_QUOTAS, EXCLUDED_ARTICLE,
                     EXCLUDED_MASTER, EXCLUDED_SUB, PRODUCTS_DIR, PROJECT_ROOT)
from app.utils.keywords import attributes_from_name
from remove_duplicates import find_duplicates

ARTICLE_TO_GROUP = {a: g for g, arts in CATEGORY_GROUPS.items() for a in arts}
MIN_SIDE = 60
KAGGLE_LICENCE = ("Kaggle 'Fashion Product Images' by Param Aggarwal (images from Myntra). "
                  "Check the licence on the Kaggle dataset page before any public/commercial use.")


def find_styles_csv(raw: Path) -> tuple[Path, Path]:
    for csv_path in sorted(raw.rglob("styles.csv")):
        img_dir = csv_path.parent / "images"
        if img_dir.is_dir():
            return csv_path, img_dir
    raise SystemExit(f"Could not find styles.csv with a sibling images/ folder under {raw}")


def read_styles(path: Path) -> tuple[list[dict], int]:
    """The Kaggle styles.csv has ~20 rows whose product name contains commas without quotes.
    We re-join the overflow into productDisplayName instead of dropping those rows."""
    rows, repaired = [], 0
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        n = len(header)
        for parts in reader:
            if not parts:
                continue
            if len(parts) > n:
                parts = parts[: n - 1] + [",".join(parts[n - 1:])]
                repaired += 1
            if len(parts) < n:
                continue
            rows.append(dict(zip(header, (p.strip() for p in parts))))
    return rows, repaired


def check_image(path: Path) -> tuple[bool, str, int, int]:
    if not path.exists():
        return False, "missing", 0, 0
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            fmt, (w, h) = im.format, im.size
            im.convert("RGB").load()
    except Exception:
        return False, "corrupt", 0, 0
    if fmt not in ("JPEG", "PNG", "WEBP"):
        return False, f"format:{fmt}", w, h
    if min(w, h) < MIN_SIDE:
        return False, "too_small", w, h
    return True, "ok", w, h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder containing styles.csv and images/")
    ap.add_argument("--target", type=int, default=1200)
    ap.add_argument("--source", default="Kaggle: paramaggarwal/fashion-product-images-small")
    ap.add_argument("--licence", default=KAGGLE_LICENCE)
    ap.add_argument("--id-prefix", default="fpi")
    ap.add_argument("--quotas", default=json.dumps(DEFAULT_QUOTAS), help="JSON {group: count}")
    ap.add_argument("--clean", action="store_true", help="empty dataset/products before copying (asks no questions)")
    args = ap.parse_args()

    raw = Path(args.raw)
    if not raw.is_absolute():
        raw = PROJECT_ROOT / raw
    styles, img_dir = find_styles_csv(raw)
    rows, repaired = read_styles(styles)
    report: dict = {"source_csv": str(styles.relative_to(PROJECT_ROOT)), "rows_read": len(rows),
                    "rows_repaired_commas": repaired, "excluded": Counter()}

    # 1) dedupe product IDs, 2) category filters
    seen, candidates = set(), []
    for r in rows:
        pid = r.get("id", "")
        if not pid or pid in seen:
            report["excluded"]["duplicate_product_id"] += 1
            continue
        seen.add(pid)
        if r.get("gender") != "Women":
            report["excluded"]["not_women"] += 1; continue
        if r.get("masterCategory") in EXCLUDED_MASTER or r.get("masterCategory") != "Apparel":
            report["excluded"][f"master:{r.get('masterCategory')}"] += 1; continue
        if r.get("subCategory") in EXCLUDED_SUB:
            report["excluded"][f"sub:{r.get('subCategory')}"] += 1; continue
        if r.get("articleType") in EXCLUDED_ARTICLE:
            report["excluded"][f"article:{r.get('articleType')}"] += 1; continue
        group = ARTICLE_TO_GROUP.get(r.get("articleType", ""))
        if not group:
            report["excluded"][f"unmapped_article:{r.get('articleType')}"] += 1; continue
        r["_group"] = group
        candidates.append(r)

    # 3) image checks
    valid = []
    for r in candidates:
        ok, why, w, h = check_image(img_dir / f"{r['id']}.jpg")
        if not ok:
            report["excluded"][f"image_{why}"] += 1; continue
        r["_w"], r["_h"] = w, h
        valid.append(r)

    # 4) duplicate images (keep the lowest product id)
    valid.sort(key=lambda r: int(r["id"]) if r["id"].isdigit() else r["id"])
    dups = find_duplicates([(r["id"], img_dir / f"{r['id']}.jpg") for r in valid])
    report["excluded"]["duplicate_image"] += len(dups)
    valid = [r for r in valid if r["id"] not in dups]

    # 5) quotas, redistributing shortfall to groups with spare products
    quotas = json.loads(args.quotas)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_group[r["_group"]].append(r)
    available = {g: len(by_group[g]) for g in quotas}
    chosen = {g: min(quotas[g], available[g]) for g in quotas}
    shortfall = {g: quotas[g] - chosen[g] for g in quotas if chosen[g] < quotas[g]}
    remaining = args.target - sum(chosen.values())
    while remaining > 0:
        spare = [g for g in quotas if available[g] > chosen[g]]
        if not spare:
            break
        for g in spare:
            if remaining == 0:
                break
            chosen[g] += 1; remaining -= 1
    selected = [r for g in quotas for r in by_group[g][: chosen[g]]]

    # 6) copy + metadata
    if args.clean and PRODUCTS_DIR.exists():
        for child in PRODUCTS_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
    out_rows = []
    for r in selected:
        rel = f"{r['_group']}/{r['id']}.jpg"   # keep the original id as filename
        dst = PRODUCTS_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_dir / f"{r['id']}.jpg", dst)
        attrs = attributes_from_name(r.get("productDisplayName", ""))
        out_rows.append({
            "product_id": f"{args.id_prefix}-{r['id']}",
            "product_name": r.get("productDisplayName") or "unknown",
            "category": r["_group"],
            "subcategory": r.get("articleType") or "unknown",
            "colour": (r.get("baseColour") or "unknown").strip() or "unknown",
            "pattern": attrs["pattern"], "sleeve_type": attrs["sleeve_type"], "neckline": attrs["neckline"],
            "image_path": rel, "source_dataset": args.source, "licence_information": args.licence,
            "gender": r.get("gender") or "unknown", "article_type": r.get("articleType") or "unknown",
            "usage": r.get("usage") or "unknown",
            "attribute_source": "colour=catalogue_metadata;pattern,sleeve_type,neckline=name_keywords",
        })
    CATALOGUE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOGUE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CATALOGUE_COLUMNS)
        w.writeheader(); w.writerows(out_rows)

    report.update({
        "excluded": dict(report["excluded"]), "candidates_after_filters": len(candidates),
        "valid_unique_images": len(valid), "available_per_group": available,
        "requested_quotas": quotas, "selected_per_group": chosen, "shortfall_vs_quota": shortfall,
        "selected_total": len(out_rows), "target": args.target,
        "meets_minimum_1000": len(out_rows) >= 1000,
    })
    (CATALOGUE_CSV.parent / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if shortfall:
        print("\nNOTE: some categories had fewer valid images than requested; the shortfall was "
              "filled from other categories. See docs/DATASET.md for legitimate ways to add more data.")


if __name__ == "__main__":
    main()
