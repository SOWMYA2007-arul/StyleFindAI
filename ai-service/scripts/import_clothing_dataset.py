"""Import a diverse subset (default 500 images) of the "Clothing dataset (full, high quality)" by
Alexey Grigorev (CC0 public domain, https://github.com/alexeygrigorev/clothing-dataset) into StyleFind.

Expected layout (folder names are detected automatically, so "archive (2)" works too):

    dataset/raw/clothing-dataset/
        images.csv              image,sender_id,label,kids
        images_original/        <image>.jpg   (full resolution, preferred for the catalogue)
        images_compressed/      <image>.jpg   (optional, 400 px wide, used for fast analysis)

What it does (nothing is deleted, existing products are never changed):
  1. reads images.csv and drops labels that are not one garment (Not sure, Skip, Other, Undershirt, Body;
     Shoes and Hat unless --include-footwear / --include-hats)
  2. checks every candidate image: exists, decodes, not too small; drops byte-identical and near-identical
     duplicates (pHash + colour thumbnail check, same rule as the rest of the project)
  3. isolates the garment (weight-free segmentation pipeline) and measures its colour and pattern
  4. picks --count images: a quota per garment type (proportional to the square root of availability, so
     rare types still appear), and inside each type a round-robin over pattern groups (solid, striped,
     checked, polka dot, printed ...) with farthest-point sampling on colour + texture, so the subset
     covers as many different patterns and colours as possible
  5. copies the chosen photos (EXIF-rotated, longest side <= --max-side) to dataset/products/<category>/,
     writes dataset/metadata/clothing_dataset_catalogue.csv, a report and a preview sheet
  6. with --load: adds them to PostgreSQL (product_code "cd-<image id>", source_dataset recorded,
     attributes stored as model_prediction / unverified because they were measured, not labelled)

    python scripts/import_clothing_dataset.py --raw ..\\dataset\\raw\\clothing-dataset --dry-run
    python scripts/import_clothing_dataset.py --raw ..\\dataset\\raw\\clothing-dataset --load
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from _common import CATALOGUE_COLUMNS, DATASET_DIR, PROJECT_ROOT, PRODUCTS_DIR

SOURCE = "clothing-dataset-full (Alexey Grigorev)"
LICENCE = "CC0 1.0 public domain - https://github.com/alexeygrigorev/clothing-dataset"
OUT_CSV = DATASET_DIR / "metadata" / "clothing_dataset_catalogue.csv"
REPORT_JSON = DATASET_DIR / "metadata" / "clothing_dataset_import_report.json"
PREVIEW = DATASET_DIR / "metadata" / "clothing_dataset_preview.jpg"
REPORT_MD = PROJECT_ROOT / "docs" / "CLOTHING_DATASET_IMPORT.md"

# dataset label -> (category slug, category display name). Existing slugs are reused.
LABEL_TO_CATEGORY = {
    "T-Shirt": ("tops-shirts", "Tops & Shirts"), "Top": ("tops-shirts", "Tops & Shirts"),
    "Longsleeve": ("tops-shirts", "Tops & Shirts"), "Polo": ("tops-shirts", "Tops & Shirts"),
    "Shirt": ("tops-shirts", "Tops & Shirts"), "Blouse": ("tops-shirts", "Tops & Shirts"),
    "Dress": ("western-dresses", "Western Dresses"),
    "Outwear": ("outerwear", "Outerwear & Jackets"), "Hoodie": ("outerwear", "Outerwear & Jackets"),
    "Blazer": ("outerwear", "Outerwear & Jackets"),
    "Pants": ("pants", "Pants & Trousers"), "Shorts": ("shorts", "Shorts"), "Skirt": ("skirts", "Skirts"),
    "Shoes": ("footwear", "Footwear"), "Hat": ("hats", "Hats & Caps"),
}
NOT_ONE_GARMENT = {"Not sure", "Skip", "Other", "Undershirt", "Body"}
LABEL_NAME = {"T-Shirt": "T-shirt", "Longsleeve": "Long-sleeve top", "Outwear": "Jacket", "Pants": "Pants"}
SLEEVE_HINT = {"T-Shirt": "short sleeves", "Polo": "short sleeves", "Longsleeve": "full sleeves"}  # project vocabulary
MIN_SIDE = 200


# ------------------------------------------------------------------ input
def find_layout(raw: Path) -> tuple[Path, Path | None, Path | None]:
    csvs = [raw / "images.csv"] + sorted(raw.rglob("images.csv"))
    csv_path = next((p for p in csvs if p.exists()), None)
    if csv_path is None:
        sys.exit(f"images.csv not found under {raw}. Put the dataset folder there (see docs/CLOTHING_DATASET.md).")
    base = csv_path.parent
    original = next((base / n for n in ("images_original", "images") if (base / n).is_dir()), None)
    compressed = base / "images_compressed" if (base / "images_compressed").is_dir() else None
    if original is None and compressed is None:
        sys.exit(f"no images_original/ or images_compressed/ folder next to {csv_path}")
    return csv_path, original, compressed


def read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    need = {"image", "label"}
    if not rows or not need <= set(rows[0]):
        sys.exit(f"{csv_path} must have the columns image,sender_id,label,kids (found {list(rows[0]) if rows else 'nothing'})")
    return rows


def image_file(folder: Path | None, image_id: str) -> Path | None:
    if folder is None:
        return None
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = folder / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def quotas(available: dict[str, int], total: int, min_each: int) -> dict[str, int]:
    """sqrt-proportional allocation, capped by availability, remainder redistributed."""
    labels = [lab for lab, n in available.items() if n > 0]
    q = {lab: 0 for lab in labels}
    left = total
    while left > 0:
        open_ = [lab for lab in labels if q[lab] < available[lab]]
        if not open_:
            break
        w = {lab: math.sqrt(available[lab]) for lab in open_}
        s = sum(w.values())
        given = 0
        for lab in open_:
            add = min(available[lab] - q[lab], max(1, int(left * w[lab] / s)))
            add = min(add, left - given)
            q[lab] += add
            given += add
            if given >= left:
                break
        left -= given
        if given == 0:
            break
    for lab in labels:  # a minimum per type where the data allows it
        if q[lab] < min(min_each, available[lab]):
            need = min(min_each, available[lab]) - q[lab]
            donor = max(q, key=lambda k: q[k])
            take = min(need, q[donor] - min_each)
            if take > 0:
                q[donor] -= take
                q[lab] += take
    return q


# ------------------------------------------------------------------ analysis
_pattern_encoder = None


def setup_pattern_model(name: str) -> str:
    """FashionCLIP zero-shot recognises floral / paisley / printed / embroidered, which the weight-free
    texture classifier leaves as 'unknown'. Used when it loads; otherwise texture only."""
    global _pattern_encoder
    if name == "none":
        return "texture classifier only (--pattern-model none)"
    try:
        from app.models.encoders import get_registry

        enc = get_registry().get(name)
        if not enc.supports_text:
            return f"texture classifier only ({name} has no text encoder)"
        _pattern_encoder = enc
        return f"texture classifier + {name} zero-shot"
    except Exception as e:  # noqa: BLE001
        return f"texture classifier only ({name} unavailable: {type(e).__name__})"


CENTRE_SHARE_MIN = 0.08   # requested colour must cover >= 8 % of the image centre to count as present


def _centre(img: Image.Image) -> Image.Image:
    """Centre band where a product shot's garment always is (rooms / walls / furniture sit at the sides)."""
    w, h = img.size
    return img.crop((int(0.3 * w), int(0.15 * h), int(0.7 * w), int(0.9 * h)))


def requested_colour_share(img: Image.Image, colour: str) -> float:
    """Share of centre pixels within CIEDE2000 18 of the requested colour (generated images only)."""
    from app.services.colour import COLOUR_REFERENCE, ciede2000, rgb_to_lab

    if colour not in COLOUR_REFERENCE:
        return 0.0
    a = np.asarray(_centre(img).resize((52, 96)), np.uint8)
    lab = rgb_to_lab(a.reshape(1, -1, 3))[0]
    ref = rgb_to_lab(np.array([[COLOUR_REFERENCE[colour]]], dtype=np.uint8))[0][0]
    return float((ciede2000(lab, ref) < 18).mean())


def analyse(path: Path, meta: dict | None = None) -> dict:
    from app.services.attributes import merge_pattern, pixel_attributes, zero_shot
    from app.services.segmentation.pipeline import isolate

    with Image.open(path) as im:
        im.load()
        img = ImageOps.exif_transpose(im).convert("RGB")
    full_size = img.size
    img.thumbnail((448, 448), Image.Resampling.LANCZOS)
    crop, info = isolate(img, None, None, "light")   # weight-free; garment on white
    attrs = pixel_attributes(crop)
    if _pattern_encoder is not None:
        zs = zero_shot(_pattern_encoder, _pattern_encoder.embed_images([crop])[0])
        attrs["pattern"] = merge_pattern(attrs["pattern"], zs["pattern"])
    colour_note = "measured on the isolated garment"
    if meta:
        # Generated product shots: the garment is centred, but rooms / walls fool the weight-free cut-out.
        share = requested_colour_share(img, meta.get("colour", ""))
        if share >= CENTRE_SHARE_MIN:
            attrs["colour"] = {"value": meta["colour"], "confidence": round(min(0.95, 0.5 + share), 3)}
            colour_note = f"requested colour confirmed in {share:.0%} of the image centre"
        else:
            centre = pixel_attributes(_centre(img))
            attrs["colour"] = centre["colour"]
            colour_note = f"requested colour not found ({share:.0%}); measured on the image centre"
        if attrs["pattern"]["value"] == "unknown" and meta.get("pattern"):
            attrs["pattern"] = {"value": meta["pattern"], "confidence": 0.5, "source": "generation prompt"}
    pal = attrs["palette"]
    lab = np.zeros(6, np.float32)
    for k, c in enumerate(pal[:2]):
        lab[3 * k:3 * k + 3] = np.asarray(c.get("lab") or c.get("lab_mean") or [0, 0, 0], np.float32) / 100.0
    pvec = np.asarray(attrs["pattern_vector"], np.float32)
    pvec = pvec / (np.linalg.norm(pvec) + 1e-6)
    return {
        "size": full_size, "colour": attrs["colour"]["value"], "secondary_colour": attrs["secondary_colour"]["value"],
        "colour_conf": float(attrs["colour"]["confidence"] or 0), "pattern": attrs["pattern"]["value"],
        "pattern_conf": float(attrs["pattern"].get("confidence") or 0), "multicolour": attrs["multicolour"]["value"],
        "seg_method": info.get("method"), "seg_conf": float(info.get("confidence") or 0),
        "seg_fallback": bool(info.get("fallback_used")), "vec": np.concatenate([lab, 0.5 * pvec]),
        "colour_note": colour_note,
    }


def pick_diverse(cands: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Round-robin over pattern groups; inside a group, the candidate farthest (colour + texture) from
    everything already chosen. Garments where segmentation fell back go last."""
    if len(cands) <= k:
        return list(cands)
    groups = defaultdict(list)
    for c in sorted(cands, key=lambda c: c["seg_fallback"]):
        groups[c["pattern"]].append(c)
    order = sorted(groups, key=lambda g: len(groups[g]))  # rare patterns first
    chosen: list[dict] = []
    vecs: list[np.ndarray] = []
    while len(chosen) < k and any(groups.values()):
        for g in order:
            if len(chosen) >= k or not groups[g]:
                continue
            pool = groups[g]
            if not vecs:
                best = rng.randrange(len(pool))
            else:
                V = np.stack(vecs)
                d = [float(np.min(np.linalg.norm(V - c["vec"], axis=1))) - (0.3 if c["seg_fallback"] else 0) for c in pool]
                best = int(np.argmax(d))
            c = pool.pop(best)
            chosen.append(c)
            vecs.append(c["vec"])
    return chosen


# ------------------------------------------------------------------ output
GENERATED_SOURCE = "AI-generated dresses (StyleFind demo)"


def colour_agrees(prompt_colour: str, measured: str) -> bool:
    """Is the measured garment colour close to the colour the prompt asked for? (CIEDE2000 between the two
    reference colours from app/services/colour.py)"""
    from app.services.colour import COLOUR_REFERENCE, ciede2000, rgb_to_lab

    if prompt_colour not in COLOUR_REFERENCE or measured not in COLOUR_REFERENCE:
        return False
    lab = rgb_to_lab(np.array([[COLOUR_REFERENCE[prompt_colour], COLOUR_REFERENCE[measured]]], dtype=np.uint8))[0]
    return float(ciede2000(lab[:1], lab[1])[0]) < 20.0


def generated_name(c: dict) -> str:
    """Name from the generation prompt (dress type, pattern, length) - but the colour word only if the
    measured colour agrees with the prompt; otherwise the measured colour is used."""
    m = c["meta"]
    colour = m["colour"] if colour_agrees(m["colour"], c["colour"]) else c["colour"]
    colour = "" if colour in ("unknown", "") else colour
    pattern = "" if m.get("pattern") in (None, "", "solid", "textured") else m["pattern"]
    length = {"knee length": "knee-length", "midi": "midi", "ankle length": "maxi"}.get(m.get("length", ""), "")
    dtype = {"a-line": "A-line", "maxi": "", "shirt dress": "shirt", "fit-and-flare": "fit-and-flare"}.get(
        m.get("dress_type", ""), m.get("dress_type", ""))
    words = [colour.capitalize(), pattern, length, dtype, "dress"]
    return " ".join(w for w in words if w)


def product_name(c: dict) -> str:
    if c.get("meta"):
        return generated_name(c)
    colour = c["colour"] if c["colour"] not in ("unknown", "") else ""
    pattern = c["pattern"] if c["pattern"] not in ("unknown", "solid", "") else ""
    base = LABEL_NAME.get(c["label"], c["label"])
    name = " ".join(w for w in (colour.capitalize(), pattern, base) if w)
    return (name + (" (kids)" if c["kids"] else "")).strip()


def save_image(src: Path, dst: Path, max_side: int) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    img.save(dst, "JPEG", quality=92, optimize=True)
    return img.size


def preview_sheet(chosen: list[dict], path: Path, cols: int = 25, cell: int = 96) -> None:
    rows = math.ceil(len(chosen) / cols)
    sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
    d = ImageDraw.Draw(sheet)
    for i, c in enumerate(chosen):
        with Image.open(c["analysis_path"]) as im:
            t = ImageOps.exif_transpose(im).convert("RGB")
        t.thumbnail((cell - 4, cell - 14))
        x, y = (i % cols) * cell, (i // cols) * cell
        sheet.paste(t, (x + (cell - t.width) // 2, y + 2))
        d.text((x + 2, y + cell - 12), f"{c['label'][:7]} {c['pattern'][:6]}", fill=(60, 60, 60))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)


TRUST_MIN = {"colour": 0.35, "pattern": 0.70}   # --trust-measured: verify only confident measurements


def attribute_rows(r: dict, trust_measured: bool) -> list[tuple[str, str, str, float | None, bool]]:
    """(attribute, value, source, confidence, verified). 'long sleeve' for the dataset's own Longsleeve label is
    human metadata; everything measured from pixels is a model prediction and stays unverified unless the
    user opts in with --trust-measured (then only above TRUST_MIN)."""
    out = []
    for attr in ("colour", "pattern", "sleeve_type", "neckline"):
        value = (r.get(attr) or "unknown").strip().lower() or "unknown"
        conf = r.get(f"_{attr}_conf")
        conf = None if conf in (None, "") else round(float(conf), 3)
        if attr == "sleeve_type" and r.get("_label") == "Longsleeve":
            out.append((attr, value, "catalogue_metadata", None, True))
            continue
        verified = bool(trust_measured and value != "unknown" and conf is not None and conf >= TRUST_MIN.get(attr, 1.1))
        out.append((attr, value, "model_prediction", conf, verified))
    return out


def load_into_db(rows: list[dict], trust_measured: bool = False, source: str = SOURCE,
                 licence: str = LICENCE) -> tuple[int, int]:
    from app.services.db import connection
    from remove_duplicates import file_sha256, image_phash

    inserted = updated = 0
    with connection() as conn:
        for slug, name in {(r["category"], r["_category_name"]) for r in rows}:
            conn.execute("INSERT INTO categories (slug, name) VALUES (%s, %s) ON CONFLICT (slug) DO NOTHING", (slug, name))
        cats = {r["slug"]: r["id"] for r in conn.execute("SELECT id, slug FROM categories")}
        for r in rows:
            path = PRODUCTS_DIR / r["image_path"]
            res = conn.execute(
                """INSERT INTO products (product_code, name, category_id, subcategory, gender, source_dataset,
                                         licence_information, description)
                   VALUES (%(code)s, %(name)s, %(cat)s, %(sub)s, %(gender)s, %(src)s, %(lic)s, %(desc)s)
                   ON CONFLICT (product_code) DO UPDATE SET name = EXCLUDED.name, category_id = EXCLUDED.category_id,
                       subcategory = EXCLUDED.subcategory, gender = EXCLUDED.gender, description = EXCLUDED.description,
                       source_dataset = EXCLUDED.source_dataset, licence_information = EXCLUDED.licence_information
                   RETURNING id, (xmax = 0) AS inserted""",
                {"code": r["product_id"], "name": r["product_name"], "cat": cats[r["category"]],
                 "sub": r["subcategory"], "gender": r["gender"] or None, "src": source, "lic": licence,
                 "desc": r["_description"]}).fetchone()
            pid = res["id"]
            inserted += res["inserted"]; updated += not res["inserted"]
            conn.execute(
                """INSERT INTO product_images (product_id, storage_key, is_primary, width, height, sha256, phash)
                   VALUES (%s, %s, TRUE, %s, %s, %s, %s)
                   ON CONFLICT (storage_key) DO UPDATE SET width = EXCLUDED.width, height = EXCLUDED.height,
                       sha256 = EXCLUDED.sha256, phash = EXCLUDED.phash""",
                (pid, r["image_path"], int(r["_w"]), int(r["_h"]), file_sha256(path), str(image_phash(path))))
            for attr, value, src, conf, verified in attribute_rows(r, trust_measured):
                conn.execute(
                    """INSERT INTO product_attributes (product_id, attribute, value, source, confidence, verified)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (product_id, attribute, source) DO UPDATE SET value = EXCLUDED.value,
                           confidence = EXCLUDED.confidence, verified = EXCLUDED.verified""",
                    (pid, attr, value, src, conf, verified))
        conn.commit()
    return inserted, updated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=DATASET_DIR / "raw" / "clothing-dataset",
                    help="folder that contains images.csv (default dataset/raw/clothing-dataset)")
    ap.add_argument("--count", type=int, default=500, help="how many images to import (default 500)")
    ap.add_argument("--min-per-type", type=int, default=10)
    ap.add_argument("--analyse-factor", type=float, default=4.0,
                    help="candidates analysed per type = factor x quota (more = more diverse, slower)")
    ap.add_argument("--kids", choices=["include", "exclude", "only"], default="include")
    ap.add_argument("--include-footwear", action="store_true", help="also import Shoes")
    ap.add_argument("--include-hats", action="store_true", help="also import Hat")
    ap.add_argument("--pattern-model", default="fashion-clip",
                    help="encoder for zero-shot pattern names (floral, paisley...); 'none' = texture only")
    ap.add_argument("--max-side", type=int, default=1024, help="resize stored photos (keeps the folder small)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="analyse and report only; copy nothing, write nothing to the DB")
    ap.add_argument("--load", action="store_true", help="also insert the products into PostgreSQL")
    ap.add_argument("--source", default=SOURCE, help="source_dataset value stored for these products")
    ap.add_argument("--licence", default=LICENCE)
    ap.add_argument("--code-prefix", default="cd", help="product codes become <prefix>-<image id>")
    ap.add_argument("--catalogue-csv", type=Path, default=OUT_CSV, help="where to write the catalogue CSV")
    ap.add_argument("--trust-measured", action="store_true",
                    help="mark confident measured colour/pattern as verified so the website shows them and filters use "
                         "them (they stay labelled model_prediction; undo with SQL in docs/CLOTHING_DATASET.md)")
    ap.add_argument("--report-dir", type=Path, default=None, help="write report + preview here instead (used by tests)")
    args = ap.parse_args()
    global REPORT_JSON, PREVIEW, REPORT_MD
    if args.code_prefix != "cd" and not args.report_dir:  # keep each import's report separate
        REPORT_JSON = DATASET_DIR / "metadata" / f"{args.code_prefix}_import_report.json"
        PREVIEW = DATASET_DIR / "metadata" / f"{args.code_prefix}_preview.jpg"
        REPORT_MD = PROJECT_ROOT / "docs" / f"IMPORT_REPORT_{args.code_prefix.upper()}.md"
    if args.report_dir:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        REPORT_JSON, PREVIEW, REPORT_MD = (args.report_dir / "report.json", args.report_dir / "preview.jpg",
                                           args.report_dir / "report.md")
        if not args.dry_run:
            raise SystemExit("--report-dir is only for --dry-run")
    t0 = time.time()
    rng = random.Random(args.seed)

    csv_path, original, compressed = find_layout(args.raw)
    rows = read_rows(csv_path)
    print(f"images.csv: {len(rows)} rows | originals: {original or '-'} | compressed: {compressed or '-'}")
    allowed = set(LABEL_TO_CATEGORY) - {"Shoes", "Hat"}
    if args.include_footwear:
        allowed.add("Shoes")
    if args.include_hats:
        allowed.add("Hat")

    report = {"source": args.source, "licence": args.licence, "csv": str(csv_path), "rows": len(rows),
              "excluded": Counter(), "rejected": Counter(), "examples": defaultdict(list)}
    seen_ids, by_label = set(), defaultdict(list)
    for r in rows:
        img_id, label = (r.get("image") or "").strip(), (r.get("label") or "").strip()
        kids = (r.get("kids") or "").strip().lower() == "true"
        if not img_id or img_id in seen_ids:
            report["rejected"]["duplicate or empty id"] += 1
            continue
        seen_ids.add(img_id)
        if label in NOT_ONE_GARMENT or label not in LABEL_TO_CATEGORY:
            report["excluded"][f"label {label or '(empty)'}"] += 1
            continue
        if label not in allowed:
            report["excluded"][f"label {label} (opt-in)"] += 1
            continue
        if (args.kids == "exclude" and kids) or (args.kids == "only" and not kids):
            report["excluded"]["kids filter"] += 1
            continue
        orig, comp = image_file(original, img_id), image_file(compressed, img_id)
        if orig is None and comp is None:
            report["rejected"]["image file missing"] += 1
            if len(report["examples"]["missing"]) < 5:
                report["examples"]["missing"].append(img_id)
            continue
        meta = {k: r[k] for k in ("dress_type", "colour", "pattern", "sleeve_type", "neckline", "length", "fabric",
                                  "presentation", "prompt", "model") if r.get(k)} if "prompt" in r else None
        by_label[label].append({"id": img_id, "label": label, "kids": kids, "sender": r.get("sender_id", ""),
                                "store_path": orig or comp, "analysis_path": comp or orig, "meta": meta})
    available = {lab: len(v) for lab, v in by_label.items()}
    q = quotas(available, args.count, args.min_per_type)
    print("pattern analysis:", setup_pattern_model(args.pattern_model))
    print("usable per type:", dict(sorted(available.items(), key=lambda kv: -kv[1])))
    print("quota per type :", dict(sorted(q.items(), key=lambda kv: -kv[1])))

    # analyse candidates (validity, duplicates, colour, pattern)
    import imagehash

    from remove_duplicates import THUMB_MAX_DIFF, file_sha256

    seen_sha: dict[str, str] = {}
    kept_hash: list[tuple[imagehash.ImageHash, np.ndarray, str]] = []
    chosen_all: list[dict] = []
    analysed = 0
    for label, cands in sorted(by_label.items(), key=lambda kv: kv[0]):
        rng.shuffle(cands)
        budget = max(q.get(label, 0), int(math.ceil(q.get(label, 0) * args.analyse_factor)))
        good = []
        for c in cands:
            if len(good) >= budget:
                break
            try:
                with Image.open(c["store_path"]) as im:
                    im.verify()
                with Image.open(c["store_path"]) as im:
                    w, h = ImageOps.exif_transpose(im).size
                if min(w, h) < MIN_SIDE:
                    report["rejected"][f"too small (< {MIN_SIDE} px)"] += 1
                    continue
                sha = file_sha256(c["store_path"])
                if sha in seen_sha:
                    report["rejected"]["byte-identical duplicate"] += 1
                    continue
                with Image.open(c["analysis_path"]) as im:
                    rgb = ImageOps.exif_transpose(im).convert("RGB")
                ph = imagehash.phash(rgb)
                th = np.asarray(rgb.resize((32, 32), Image.Resampling.BILINEAR), np.float32)
                if any(ph - kh <= 3 and float(np.abs(th - kt).mean()) <= THUMB_MAX_DIFF for kh, kt, _ in kept_hash):
                    report["rejected"]["near-duplicate photo"] += 1
                    continue
                a = analyse(c["analysis_path"], c.get("meta"))
                analysed += 1
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
                report["rejected"]["corrupt / unreadable"] += 1
                if len(report["examples"]["corrupt"]) < 5:
                    report["examples"]["corrupt"].append(f"{c['id']}: {type(e).__name__}")
                continue
            seen_sha[sha] = c["id"]
            kept_hash.append((ph, th, c["id"]))
            c.update(a, sha=sha, width=w, height=h)
            good.append(c)
            if analysed % 100 == 0:
                print(f"  analysed {analysed} images ({time.time() - t0:.0f} s)", flush=True)
        picked = pick_diverse(good, q.get(label, 0), rng)
        chosen_all += picked
        print(f"  {label:<11} quota {q.get(label, 0):>3}  valid analysed {len(good):>4}  picked {len(picked):>3}  "
              f"patterns {dict(Counter(c['pattern'] for c in picked))}")

    # fill a shortfall (e.g. some files missing) from types that still have unused candidates is not attempted
    # on purpose: the report shows it instead of silently skewing the mix.
    out_rows = []
    for c in chosen_all:
        slug, cat_name = LABEL_TO_CATEGORY[c["label"]]
        rel = f"{slug}/{args.code_prefix}-{c['id']}.jpg"
        m = c.get("meta")
        if m:
            agrees = c["colour"] == m.get("colour") or colour_agrees(m.get("colour", ""), c["colour"])
            c["colour_agrees"] = agrees
            desc = (f"AI-GENERATED product image (not a real product), made with {m.get('model', 'Stable Diffusion')}. "
                    f"Requested design: {m.get('prompt', '')[:400]}. Colour: {c.get('colour_note', '')}.")
            sub = (m.get("dress_type") or c["label"]).replace("-", " ").title()
        else:
            desc = (f"{c['label']} from the public-domain Clothing Dataset (contributor {c['sender']}). "
                    f"Colour and pattern were measured from the photo, not labelled.")
            sub = c["label"]
        out_rows.append({
            "product_id": f"{args.code_prefix}-{c['id']}", "product_name": product_name(c), "category": slug,
            "subcategory": sub, "colour": c["colour"], "pattern": c["pattern"],
            "sleeve_type": (m or {}).get("sleeve_type") or SLEEVE_HINT.get(c["label"], "unknown"),
            "neckline": (m or {}).get("neckline") or "unknown", "image_path": rel,
            "source_dataset": args.source, "licence_information": args.licence, "gender": "kids" if c["kids"] else "",
            "article_type": c["label"], "usage": "",
            "attribute_source": ("design=generation prompt (unverified); colour/pattern=pixels" if m
                                 else "label=dataset; colour/pattern=pixels"),
            "_category_name": cat_name, "_colour_conf": c["colour_conf"], "_pattern_conf": c["pattern_conf"],
            "_sleeve_type_conf": 0.5 if m else (0.9 if c["label"] in SLEEVE_HINT else None),
            "_neckline_conf": 0.5 if m else None, "_label": c["label"],
            "_description": desc, "_src": c["store_path"], "_w": c["width"], "_h": c["height"]})

    stats = {
        "selected": len(out_rows), "requested": args.count, "analysed": analysed,
        "per_type": dict(Counter(r["subcategory"] for r in out_rows)),
        "per_category": dict(Counter(r["category"] for r in out_rows)),
        "per_pattern": dict(Counter(r["pattern"] for r in out_rows)),
        "per_colour": dict(Counter(r["colour"] for r in out_rows)),
        "kids": sum(1 for r in out_rows if r["gender"] == "kids"),
        "distinct_type_pattern_colour": len({(r["subcategory"], r["pattern"], r["colour"]) for r in out_rows}),
        "segmentation_fallbacks": sum(1 for c in chosen_all if c["seg_fallback"]),
        "distinct_contributors": len({c["sender"] for c in chosen_all}),
        "generated_colour_agreement": (round(sum(c.get("colour_agrees", False) for c in chosen_all) /
                                             max(1, sum(1 for c in chosen_all if c.get("meta"))), 3)
                                       if any(c.get("meta") for c in chosen_all) else None),
    }
    report.update(stats, excluded=dict(report["excluded"]), rejected=dict(report["rejected"]),
                  examples=dict(report["examples"]), quota=q, available=available, seed=args.seed,
                  dry_run=args.dry_run, seconds=round(time.time() - t0, 1))
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2))
    preview_sheet(chosen_all, PREVIEW)
    write_md(report)
    print(f"\nselected {len(out_rows)} of {args.count} requested | patterns {stats['per_pattern']}")
    print(f"report: {REPORT_MD}\npreview: {PREVIEW}")
    if len(out_rows) < args.count:
        print(f"NOTE: only {len(out_rows)} usable images were found (see 'rejected' / 'excluded' in the report).")
    if args.dry_run:
        print("dry run: nothing copied, nothing written to the database.")
        return

    for i, r in enumerate(out_rows):
        r["_w"], r["_h"] = save_image(r["_src"], PRODUCTS_DIR / r["image_path"], args.max_side)
    with open(args.catalogue_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CATALOGUE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"copied {len(out_rows)} photos to {PRODUCTS_DIR} and wrote {args.catalogue_csv}")
    if args.load:
        ins, upd = load_into_db(out_rows, args.trust_measured, args.source, args.licence)
        print(f"database: {ins} products added, {upd} updated (existing products untouched).")
        print("next: python scripts/generate_embeddings.py --model fashion-clip   (embeds only the new images)")
    else:
        print("not loaded into the database yet: re-run with --load")


def write_md(r: dict) -> None:
    def table(d: dict, head: str) -> list[str]:
        out = [f"| {head} | images |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in sorted(d.items(), key=lambda kv: -kv[1])]
        return out

    L = [f"# Import report: {r['source']}", "",
         f"Source: {r['source']}  ", f"Licence: {r['licence']}  ",
         f"images.csv rows: {r['rows']} | analysed: {r['analysed']} | selected: **{r['selected']}** of {r['requested']} "
         f"| kids items: {r['kids']} | contributors: {r['distinct_contributors']} | "
         f"distinct type+pattern+colour combinations: {r['distinct_type_pattern_colour']} | "
         f"segmentation fallbacks among selected: {r['segmentation_fallbacks']} | {'DRY RUN' if r['dry_run'] else 'files copied'}",
         "", "Colour and pattern are measured from pixels on the isolated garment (unverified, stored as "
         "`model_prediction`); the garment type comes from the dataset's human label.", "",
         "## Per garment type (quota / usable)", "", "| type | quota | usable in folder | selected |", "|---|---|---|---|"]
    for lab in sorted(r["available"], key=lambda k: -r["available"][k]):
        L.append(f"| {lab} | {r['quota'].get(lab, 0)} | {r['available'][lab]} | {r['per_type'].get(lab, 0)} |")
    L += ["", "## Per category", ""] + table(r["per_category"], "category")
    L += ["", "## Patterns", ""] + table(r["per_pattern"], "pattern")
    L += ["", "## Colours", ""] + table(r["per_colour"], "colour")
    L += ["", "## Excluded (by rule)", ""] + (table(r["excluded"], "reason") if r["excluded"] else ["none"])
    L += ["", "## Rejected (by checks)", ""] + (table(r["rejected"], "reason") if r["rejected"] else ["none"])
    if r.get("generated_colour_agreement") is not None:
        L += ["", f"Measured colour agrees with the requested colour for **{r['generated_colour_agreement']:.0%}** of the "
                  "selected generated images (product names use the measured colour where they disagree)."]
    L += ["", f"Preview of the selection: `dataset/metadata/{PREVIEW.name}`."]
    REPORT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
