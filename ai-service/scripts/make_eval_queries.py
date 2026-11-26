"""Create evaluation queries with KNOWN answers from catalogue images.

For a sample of catalogue products we synthesise "customer-like" photos:
    background  - garment pasted onto a coloured / textured background (same product, different background)
    angle       - rotation + perspective warp (same product, different angle)
    partial     - only part of the garment visible
    clutter     - extra objects in the photo (multiple objects)
    lighting    - darker / warmer, slight blur and JPEG compression (phone-photo look)
Ground truth = the source product. These are augmentations, NOT real customer photos; for
true street-to-shop evaluation add a query CSV built from DeepFashion2 or your own photos
(format: query_path,product_id,transform) and pass it to evaluate.py --queries.

    python scripts/make_eval_queries.py --n 60
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from _common import CATALOGUE_CSV, DATASET_DIR, PRODUCTS_DIR
from app.services.foreground import foreground_mask

OUT = DATASET_DIR / "evaluation" / "queries"
TRANSFORMS = ["background", "angle", "partial", "clutter", "lighting"]
EXTRA_TRANSFORMS = ["occlusion"]  # opt-in: adding it changes the query set (old baselines stay comparable)


def _garment_rgba(img: Image.Image) -> Image.Image:
    rgb, mask = foreground_mask(img, size=max(img.size))
    m = Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size).filter(ImageFilter.GaussianBlur(1.5))
    out = img.convert("RGBA"); out.putalpha(m)
    return out


def _background(size, rng):
    w, h = size
    base = tuple(rng.randint(60, 200) for _ in range(3))
    bg = Image.new("RGB", (w, h), base)
    d = ImageDraw.Draw(bg)
    for _ in range(rng.randint(3, 8)):   # soft room-like shapes
        c = tuple(int(np.clip(v + rng.randint(-50, 50), 0, 255)) for v in base)
        x, y = rng.randint(0, w), rng.randint(0, h)
        d.rectangle((x, y, x + rng.randint(40, w), y + rng.randint(40, h)), fill=c)
    noise = np.random.default_rng(rng.randint(0, 9999)).normal(0, 8, (h, w, 3))
    return Image.fromarray(np.clip(np.asarray(bg) + noise, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))


def transform(img: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if kind == "background":
        g = _garment_rgba(img)
        bg = _background((int(w * 1.3), int(h * 1.2)), rng)
        bg.paste(g, (int(w * rng.uniform(0.05, 0.25)), int(h * rng.uniform(0.02, 0.15))), g)
        return bg
    if kind == "angle":
        out = img.rotate(rng.uniform(-18, 18), resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(235, 235, 230))
        ow, oh = out.size
        dx = ow * rng.uniform(0.04, 0.10)
        coeffs = _perspective([(0, 0), (ow, 0), (ow, oh), (0, oh)], [(dx, 0), (ow - dx, dx), (ow, oh), (0, oh - dx)])
        return out.transform(out.size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC, fillcolor=(235, 235, 230))
    if kind == "partial":
        fx, fy = rng.uniform(0.55, 0.75), rng.uniform(0.5, 0.7)
        x0 = rng.randint(0, int(w * (1 - fx))); y0 = rng.randint(0, int(h * (1 - fy)))
        return img.crop((x0, y0, x0 + int(w * fx), y0 + int(h * fy))).resize((w, h))
    if kind == "clutter":
        canvas = Image.new("RGB", (int(w * 1.6), int(h * 1.2)), tuple(rng.randint(200, 245) for _ in range(3)))
        canvas.paste(img, (int(w * 0.05), int(h * 0.1)))
        d = ImageDraw.Draw(canvas)
        for _ in range(rng.randint(2, 4)):  # bag / shoe / jewellery-like blobs
            x, y = rng.randint(int(w * 1.0), int(w * 1.45)), rng.randint(0, int(h * 1.0))
            s = rng.randint(30, 80)
            d.ellipse((x, y, x + s, y + int(s * rng.uniform(0.6, 1.4))),
                      fill=tuple(rng.randint(0, 255) for _ in range(3)))
        return canvas
    if kind == "occlusion":  # e.g. a hand, bag or hair covering part of the garment
        out = img.copy()
        d = ImageDraw.Draw(out)
        ow, oh = int(w * rng.uniform(0.2, 0.35)), int(h * rng.uniform(0.15, 0.3))
        x0, y0 = rng.randint(0, w - ow), rng.randint(int(h * 0.2), int(h * 0.7))
        d.rounded_rectangle((x0, y0, x0 + ow, y0 + oh), radius=min(ow, oh) // 3,
                            fill=tuple(rng.randint(120, 210) for _ in range(3)))
        return out
    if kind == "lighting":
        out = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 0.85))
        r, g, b = out.split()
        out = Image.merge("RGB", (r.point(lambda v: min(255, int(v * 1.08))), g, b.point(lambda v: int(v * 0.9))))
        return out.filter(ImageFilter.GaussianBlur(1.2))
    raise ValueError(kind)


def _perspective(src, dst):
    m = []
    for (x, y), (u, v) in zip(dst, src):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); m.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(m, dtype=float); b = np.array(src, dtype=float).reshape(8)
    return np.linalg.solve(a, b).tolist()


def split_for(product_id: str) -> str:
    """Product-level val/test split (same rule as evaluate.py) so tuning never sees test products."""
    import zlib

    return "val" if zlib.crc32(product_id.encode()) % 2 == 0 else "test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="number of source products")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--extra-transforms", nargs="*", default=[], choices=EXTRA_TRANSFORMS)
    ap.add_argument("--catalogue", type=Path, default=CATALOGUE_CSV,
                    help="catalogue CSV to sample from (e.g. dataset/metadata/clothing_dataset_catalogue.csv)")
    ap.add_argument("--name", default="queries",
                    help="output name: dataset/evaluation/<name>/ + <name>.csv (default 'queries' = the baseline set)")
    args = ap.parse_args()
    OUT = DATASET_DIR / "evaluation" / args.name
    with open(args.catalogue, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    if OUT.exists():
        # keep a copy of the previous query set instead of deleting it (baselines refer to it)
        backup = OUT.parent / f"{args.name}_backup_{__import__('time').strftime('%Y%m%d-%H%M%S')}"
        shutil.move(str(OUT), backup)
        csv_old = DATASET_DIR / "evaluation" / f"{args.name}.csv"
        if csv_old.exists():
            shutil.copy(csv_old, backup / f"{args.name}.csv")
        print(f"previous queries moved to {backup}")
    OUT.mkdir(parents=True)
    out_rows = []
    for r in sample:
        with Image.open(PRODUCTS_DIR / r["image_path"]) as im:
            src = im.convert("RGB")
        if min(src.size) < 240:  # tiny catalogue thumbnails (Kaggle small = 60x80): simulate a phone-sized photo
            f = 240 / min(src.size)
            src = src.resize((round(src.width * f), round(src.height * f)), Image.Resampling.BICUBIC)
        for kind in TRANSFORMS + args.extra_transforms:
            name = f"{r['product_id']}__{kind}.jpg"
            transform(src, kind, rng).save(OUT / name, quality=85)
            out_rows.append({"query_path": f"{args.name}/{name}", "product_id": r["product_id"], "transform": kind,
                             "category": r["category"], "split": split_for(r["product_id"])})
    with open(DATASET_DIR / "evaluation" / f"{args.name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    print(f"Wrote {len(out_rows)} queries for {len(sample)} products to {OUT}")


if __name__ == "__main__":
    main()
