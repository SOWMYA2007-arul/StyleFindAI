"""READ-ONLY inspection of a new image dataset before any integration decision.

Nothing is written to the database, no images are copied or modified. Output:
docs/NEW_DATASET_REPORT.md and dataset/metadata/new_dataset_report.json

    python scripts/inspect_dataset.py --root "D:\\Datasets\\my-new-dataset"
    python scripts/inspect_dataset.py --root ... --category-column articleType --max-images 3000
    python scripts/inspect_dataset.py --root ... --full          # check every image (slower)

Reports: folder layout, metadata files (CSV/JSON columns + row counts), image formats, resolution
distribution (vs the current 60x80 catalogue), aspect ratios, corrupt files, exact + near
duplicates, category distribution and how it maps onto StyleFind's six categories, likely
women's-clothing share, and licence / README text found in the dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from _common import CATALOGUE_CSV, CATEGORY_GROUPS, PRODUCTS_DIR, PROJECT_ROOT
from remove_duplicates import find_duplicates

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
LICENCE_HINTS = ("license", "licence", "readme", "dataset-metadata", "datacard", "citation", "terms")
CATEGORY_COLUMN_GUESSES = ("articletype", "article_type", "category", "subcategory", "sub_category", "class",
                           "label", "product_type", "type", "masterCategory".lower())
KEYWORD_GROUPS = {  # extra keywords to map arbitrary category names onto StyleFind groups
    "kurtas-kurtis": ["kurta", "kurti", "tunic", "anarkali"], "sarees": ["saree", "sari"],
    "lehengas": ["lehenga", "lehnga", "ghagra", "chaniya"], "western-dresses": ["dress", "gown", "jumpsuit", "frock"],
    "tops-shirts": ["top", "shirt", "blouse", "tee", "t-shirt", "tshirt"],
    "other-ethnic": ["salwar", "churidar", "dupatta", "palazzo", "sharara", "patiala", "ethnic"],
}
EXCLUDE_WORDS = ("shoe", "sandal", "heel", "bag", "wallet", "watch", "jewel", "earring", "necklace", "bra", "brief",
                 "night", "lingerie", "sock", "belt", "cap", "hat", "sunglass", "perfume", "lipstick")


def map_category(name: str) -> str | None:
    n = (name or "").lower()
    for g, arts in CATEGORY_GROUPS.items():
        if n in (a.lower() for a in arts):
            return g
    if any(w in n for w in EXCLUDE_WORDS):
        return None
    for g, words in KEYWORD_GROUPS.items():
        if any(re.search(rf"\b{re.escape(w)}", n) for w in words):
            return g
    return None


def scan(root: Path):
    images, meta, licence = [], [], []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXT:
            images.append(p)
        elif ext in (".csv", ".json", ".jsonl", ".tsv", ".txt", ".md", ".parquet") or not ext:
            low = p.name.lower()
            if any(h in low for h in LICENCE_HINTS):
                licence.append(p)
            if ext in (".csv", ".tsv", ".json", ".jsonl", ".parquet"):
                meta.append(p)
    return images, meta, licence


def read_table(path: Path, max_rows: int = 200000) -> tuple[list[str], list[dict]]:
    if path.suffix.lower() in (".csv", ".tsv"):
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            sample = f.read(4096); f.seek(0)
            delim = "\t" if path.suffix.lower() == ".tsv" or sample.count("\t") > sample.count(",") else ","
            reader = csv.reader(f, delimiter=delim)
            header = next(reader, [])
            rows = []
            for i, parts in enumerate(reader):
                if i >= max_rows:
                    break
                if len(parts) > len(header):  # unquoted commas in the last field (Kaggle styles.csv)
                    parts = parts[: len(header) - 1] + [",".join(parts[len(header) - 1:])]
                rows.append(dict(zip(header, parts)))
            return header, rows
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(l) for l in open(path, encoding="utf-8", errors="replace") if l.strip()][:max_rows]
        return (list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []), [r for r in rows if isinstance(r, dict)]
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            return [], []
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return list(data[0].keys()), data[:max_rows]
        if isinstance(data, dict):
            return list(data.keys()), []
    return [], []


def percentiles(v):
    if not v:
        return {}
    a = np.array(v)
    return {k: int(np.percentile(a, q)) for k, q in (("min", 0), ("p10", 10), ("median", 50), ("p90", 90), ("max", 100))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--category-column", default=None)
    ap.add_argument("--max-images", type=int, default=3000, help="sample size for image checks (deterministic)")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a folder: {root}")
    print(f"scanning {root} (read-only) ...")
    images, meta_files, licence_files = scan(root)
    rep: dict = {"root": str(root), "total_image_files": len(images),
                 "extensions": dict(Counter(p.suffix.lower() for p in images).most_common())}

    # ---------- metadata tables
    tables = []
    best_cat = None
    for m in sorted(meta_files, key=lambda p: p.stat().st_size, reverse=True)[:10]:
        header, rows = read_table(m)
        info = {"file": str(m.relative_to(root)), "rows_read": len(rows), "columns": header[:40]}
        small = {}
        for col in header[:40]:
            vals = Counter(str(r.get(col, "")).strip() for r in rows)
            if 1 < len(vals) <= 60:
                small[col] = dict(vals.most_common(25))
        info["low_cardinality_columns"] = small
        tables.append(info)
        cols = {c.lower(): c for c in header}
        cand = args.category_column if args.category_column in header else next(
            (cols[g] for g in CATEGORY_COLUMN_GUESSES if g in cols), None)
        if cand and rows and best_cat is None:
            best_cat = (m, cand, rows, header)
    rep["metadata_tables"] = tables

    # ---------- categories
    if best_cat:
        m, col, rows, header = best_cat
        counts = Counter(str(r.get(col, "")).strip() for r in rows)
        gender_col = next((c for c in header if c.lower() in ("gender", "sex", "target")), None)
        women = Counter(str(r.get(col, "")).strip() for r in rows
                        if gender_col and str(r.get(gender_col, "")).lower() in ("women", "woman", "female", "girls", "f"))
        rep["category_source"] = f"column '{col}' in {m.relative_to(root)}"
        rep["gender_column"] = gender_col
        rep["categories_top"] = dict(counts.most_common(60))
        src = women if gender_col else counts
    else:
        counts = Counter(p.parent.name for p in images)
        rep["category_source"] = "parent folder names (no metadata column found; pass --category-column)"
        rep["categories_top"] = dict(counts.most_common(60))
        rep["gender_column"] = None
        src = counts
    mapped = Counter()
    for name, n in src.items():
        g = map_category(name)
        mapped[g or "not mapped / excluded"] += n
    rep["stylefind_mapping"] = dict(mapped.most_common())
    rep["stylefind_mapping_basis"] = "women only" if rep.get("gender_column") else "all rows (no gender column)"

    # ---------- image checks (sample)
    rng = random.Random(args.seed)
    sample = images if args.full or len(images) <= args.max_images else rng.sample(images, args.max_images)
    widths, heights, aspects, corrupt, modes = [], [], [], [], Counter()
    ok_paths = []
    for p in sample:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
                modes[im.mode] += 1
            widths.append(w); heights.append(h); aspects.append(round(h / max(1, w), 2)); ok_paths.append(p)
        except Exception as e:  # noqa: BLE001
            corrupt.append(f"{p.relative_to(root)} ({type(e).__name__})")
    short = [min(w, h) for w, h in zip(widths, heights)]
    rep["image_sample_size"] = len(sample)
    rep["corrupt_in_sample"] = len(corrupt); rep["corrupt_examples"] = corrupt[:20]
    rep["width"] = percentiles(widths); rep["height"] = percentiles(heights)
    rep["aspect_h_over_w"] = percentiles([a * 100 for a in aspects])
    rep["share_short_side_ge_224"] = round(float(np.mean([s >= 224 for s in short])), 3) if short else 0
    rep["share_short_side_ge_512"] = round(float(np.mean([s >= 512 for s in short])), 3) if short else 0
    rep["colour_modes"] = dict(modes)
    dup_sample = ok_paths[: min(len(ok_paths), 3000)]
    dups = find_duplicates([(str(p.relative_to(root)), p) for p in dup_sample])
    rep["duplicates_in_checked"] = {"checked": len(dup_sample), "duplicates": len(dups),
                                    "examples": [f"{a} == {b}" for a, b in list(dups.items())[:15]]}
    # current catalogue resolution, for comparison
    cur = []
    if CATALOGUE_CSV.exists():
        with open(CATALOGUE_CSV, newline="", encoding="utf-8") as f:
            for r in list(csv.DictReader(f))[:300]:
                try:
                    with Image.open(PRODUCTS_DIR / r["image_path"]) as im:
                        cur.append(min(im.size))
                except Exception:  # noqa: BLE001
                    pass
    rep["current_catalogue_short_side"] = percentiles(cur)

    # ---------- licence text
    lic = []
    for p in licence_files[:10]:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits = re.findall(r"(?i)(cc[- ]by[^\n]{0,40}|cc0[^\n]{0,30}|mit licen[cs]e|apache[^\n]{0,20}|"
                          r"non[- ]?commercial[^\n]{0,60}|research purposes[^\n]{0,60}|all rights reserved|"
                          r"\"license[s]?\"\s*:\s*[^\n]{0,80})", txt)
        lic.append({"file": str(p.relative_to(root)), "first_chars": txt[:500], "licence_mentions": hits[:10]})
    rep["licence_files"] = lic
    rep["licence_verdict"] = ("licence text found - READ IT before integrating" if lic else
                              "NO licence file found - check the download page; do not publish images without a licence")

    out_json = PROJECT_ROOT / "dataset" / "metadata" / "new_dataset_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, indent=2, default=str))
    L = ["# New dataset inspection (read-only)", "", f"Folder: `{root}`", "",
         f"* Image files: **{rep['total_image_files']}** {rep['extensions']}",
         f"* Checked sample: {rep['image_sample_size']} images, corrupt: {rep['corrupt_in_sample']}",
         f"* Width {rep['width']}, height {rep['height']}",
         f"* Short side >= 224 px: {rep['share_short_side_ge_224']:.0%}, >= 512 px: {rep['share_short_side_ge_512']:.0%} "
         f"(current catalogue short side: {rep['current_catalogue_short_side']})",
         f"* Duplicates among {rep['duplicates_in_checked']['checked']} checked: {rep['duplicates_in_checked']['duplicates']}",
         f"* Categories from: {rep['category_source']}", "",
         f"## Mapping to StyleFind categories ({rep['stylefind_mapping_basis']})", "", "| StyleFind category | images |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in rep["stylefind_mapping"].items()]
    L += ["", "## Top source categories", "", "| category | count |", "|---|---|"]
    L += [f"| {k or '(empty)'} | {v} |" for k, v in list(rep["categories_top"].items())[:30]]
    L += ["", "## Metadata files", ""] + [f"* `{t['file']}`: {t['rows_read']} rows, columns {t['columns'][:15]}" for t in tables]
    L += ["", "## Licence", "", f"**{rep['licence_verdict']}**", ""]
    for x in lic:
        L += [f"### `{x['file']}`", "", f"Mentions: {x['licence_mentions'] or 'none detected'}", "", "```",
              x["first_chars"].replace("```", "'''"), "```", ""]
    if corrupt:
        L += ["## Corrupt examples", ""] + [f"* {c}" for c in corrupt[:20]]
    L += ["", "## Next step", "", "Nothing has been imported. If the licence allows use and the mapping looks right, the",
          "integration plan is in docs/DATASET.md (separate catalogue version, embeddings for the new images,",
          "evaluation against the saved FashionCLIP baseline, then your approval before switching)."]
    (PROJECT_ROOT / "docs" / "NEW_DATASET_REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:30]))
    print(f"\nfull report: docs/NEW_DATASET_REPORT.md and {out_json}")


if __name__ == "__main__":
    main()
