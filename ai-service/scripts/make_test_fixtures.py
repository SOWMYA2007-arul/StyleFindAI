"""Generate a SYNTHETIC TEST FIXTURE in the Kaggle 'Fashion Product Images' layout.

    *** THIS IS NOT THE PRODUCT CATALOGUE. ***

It exists so the dataset scripts, database, AI search and website can be tested end-to-end
on a machine that cannot download the Kaggle dataset. Every row is tagged
source_dataset = "SYNTHETIC_TEST_FIXTURE". For the real website, download the Kaggle dataset
(see README) and run prepare_dataset.py on it instead.

The fixture deliberately includes junk rows (footwear, innerwear, a missing image, a broken
image, a duplicate image under a new ID, a duplicate product ID) so the cleaning
steps can be tested.

Usage:  python scripts/make_test_fixtures.py --per-type 25
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from _common import DATASET_DIR

OUT = DATASET_DIR / "raw" / "synthetic-fixture"

ARTICLES = {  # articleType -> (silhouette, usage)
    "Kurtas": ("kurta", "Ethnic"), "Kurtis": ("kurta", "Ethnic"), "Sarees": ("saree", "Ethnic"),
    "Lehenga Choli": ("lehenga", "Ethnic"), "Dresses": ("dress", "Casual"), "Tops": ("top", "Casual"),
    "Shirts": ("shirt", "Formal"), "Churidar": ("pants", "Ethnic"), "Dupatta": ("dupatta", "Ethnic"),
}
COLOURS = {"Red": (200, 30, 40), "Navy Blue": (25, 30, 80), "Blue": (40, 90, 200), "Pink": (245, 160, 190),
           "Green": (40, 140, 60), "Yellow": (245, 220, 50), "Black": (25, 25, 25), "White": (245, 245, 240),
           "Maroon": (110, 15, 30), "Purple": (110, 50, 140), "Orange": (240, 120, 30), "Teal": (0, 120, 120)}
PATTERNS = ["Solid", "Striped", "Checked", "Polka", "Floral", "Printed"]
SLEEVES = ["Sleeveless", "Short Sleeves", "3/4 Sleeves", "Full Sleeves"]
NECKS = ["Round Neck", "V-Neck", "Mandarin Collar", "Boat Neck"]


def silhouette(kind: str, w: int, h: int, rng: random.Random, sleeve: str) -> list[tuple]:
    cx = w / 2
    s = rng.uniform(0.9, 1.1)
    sw = {"Sleeveless": 0, "Short Sleeves": 0.12, "3/4 Sleeves": 0.25, "Full Sleeves": 0.38}[sleeve]
    if kind in ("kurta", "dress", "top", "shirt"):
        length = {"kurta": 0.82, "dress": 0.78, "top": 0.45, "shirt": 0.5}[kind]
        flare = {"kurta": 0.30, "dress": 0.40, "top": 0.22, "shirt": 0.22}[kind] * s
        sh = 0.20 * w
        top, bot = 0.1 * h, (0.1 + length) * h
        pts = [(cx - 0.05 * w, top), (cx - sh, top + 0.03 * h)]
        if sw:
            pts += [(cx - sh - 0.08 * w, top + 0.03 * h + sw * h), (cx - sh + 0.02 * w, top + 0.06 * h + sw * h)]
        pts += [(cx - 0.17 * w, top + 0.25 * h), (cx - flare * w, bot), (cx + flare * w, bot),
                (cx + 0.17 * w, top + 0.25 * h)]
        if sw:
            pts += [(cx + sh - 0.02 * w, top + 0.06 * h + sw * h), (cx + sh + 0.08 * w, top + 0.03 * h + sw * h)]
        pts += [(cx + sh, top + 0.03 * h), (cx + 0.05 * w, top)]
        return pts
    if kind == "lehenga":
        return [(cx - 0.12 * w, 0.12 * h), (cx + 0.12 * w, 0.12 * h), (cx + 0.14 * w, 0.30 * h),
                (cx + 0.42 * w * s, 0.92 * h), (cx - 0.42 * w * s, 0.92 * h), (cx - 0.14 * w, 0.30 * h)]
    if kind == "saree":
        return [(cx - 0.25 * w, 0.08 * h), (cx + 0.15 * w, 0.08 * h), (cx + 0.32 * w, 0.92 * h),
                (cx - 0.32 * w, 0.92 * h), (cx - 0.1 * w, 0.5 * h)]
    if kind == "pants":
        return [(cx - 0.18 * w, 0.08 * h), (cx + 0.18 * w, 0.08 * h), (cx + 0.16 * w, 0.92 * h),
                (cx + 0.03 * w, 0.92 * h), (cx, 0.3 * h), (cx - 0.03 * w, 0.92 * h), (cx - 0.16 * w, 0.92 * h)]
    # dupatta: long rectangle draped diagonally
    return [(cx - 0.3 * w, 0.1 * h), (cx - 0.1 * w, 0.06 * h), (cx + 0.32 * w, 0.9 * h), (cx + 0.12 * w, 0.94 * h)]


def texture(w: int, h: int, base: tuple, pattern: str, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (w, h), base)
    d = ImageDraw.Draw(img)
    accent = tuple(int(255 - c * 0.8) if sum(base) < 380 else int(c * 0.35) for c in base)
    if rng.random() < 0.5:
        accent = rng.choice(list(COLOURS.values()))
    step = rng.randint(14, 30)
    if pattern == "Striped":
        horiz = rng.random() < 0.5
        for i in range(0, w + h, step):
            d.rectangle((0, i, w, i + step // 3) if horiz else (i, 0, i + step // 3, h), fill=accent)
    elif pattern == "Checked":
        for i in range(0, max(w, h), step):
            d.rectangle((i, 0, i + step // 4, h), fill=accent)
            d.rectangle((0, i, w, i + step // 4), fill=accent)
    elif pattern == "Polka":
        for y in range(0, h, step):
            for x in range((y // step % 2) * step // 2, w, step):
                d.ellipse((x, y, x + step // 3, y + step // 3), fill=accent)
    elif pattern == "Floral":
        for _ in range(int(w * h / (step * step * 1.2))):
            x, y = rng.randint(0, w), rng.randint(0, h)
            r = rng.randint(4, 9)
            for a in range(5):
                ang = a * 2 * np.pi / 5
                d.ellipse((x + r * np.cos(ang) - r / 2, y + r * np.sin(ang) - r / 2,
                           x + r * np.cos(ang) + r / 2, y + r * np.sin(ang) + r / 2), fill=accent)
            d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(250, 220, 60))
    elif pattern == "Printed":
        for _ in range(int(w * h / (step * step * 0.8))):
            x, y = rng.randint(0, w), rng.randint(0, h)
            s = rng.randint(3, 10)
            d.polygon([(x, y - s), (x + s, y), (x, y + s), (x - s, y)], fill=accent)
    return img


def render(article: str, colour: str, pattern: str, sleeve: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    w, h = 300, 400
    kind = ARTICLES[article][0]
    bg = tuple(rng.randint(238, 252) for _ in range(3))
    canvas = Image.new("RGB", (w, h), bg)
    base = tuple(int(np.clip(c + rng.randint(-15, 15), 0, 255)) for c in COLOURS[colour])
    tex = texture(w, h, base, pattern, rng)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(silhouette(kind, w, h, rng, sleeve), fill=255)
    canvas.paste(tex, (0, 0), mask)
    if kind in ("lehenga", "saree"):  # border trim
        ImageDraw.Draw(canvas).line([(w * 0.1, h * 0.9), (w * 0.9, h * 0.9)], fill=(200, 160, 60), width=8)
    return canvas.filter(ImageFilter.GaussianBlur(0.6))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)  # only ever removes the synthetic fixture folder
    (OUT / "images").mkdir(parents=True)
    rng = random.Random(args.seed)
    rows = []
    pid = 900000
    for article in ARTICLES:
        for i in range(args.per_type):
            pid += 1
            colour, pattern = rng.choice(list(COLOURS)), rng.choice(PATTERNS)
            sleeve, neck = rng.choice(SLEEVES), rng.choice(NECKS)
            render(article, colour, pattern, sleeve, pid).save(OUT / "images" / f"{pid}.jpg", quality=92)
            name = f"SYNTHETIC Women {colour} {pattern} {article.rstrip('s')} with {sleeve} and {neck}"
            rows.append([pid, "Women", "Apparel", "Topwear", article, colour, "Summer", 2024,
                         ARTICLES[article][1], name])
    # --- deliberate junk rows for the cleaning tests ---
    pid += 1; render("Tops", "Red", "Solid", "Short Sleeves", 1).save(OUT / "images" / f"{pid}.jpg")
    rows.append([pid, "Women", "Footwear", "Shoes", "Heels", "Red", "Summer", 2024, "Casual", "SYNTHETIC Red Heels"])
    pid += 1; render("Tops", "Black", "Solid", "Sleeveless", 2).save(OUT / "images" / f"{pid}.jpg")
    rows.append([pid, "Women", "Apparel", "Innerwear", "Bra", "Black", "Summer", 2024, "Casual", "SYNTHETIC Bra"])
    pid += 1  # missing image
    rows.append([pid, "Women", "Apparel", "Topwear", "Tops", "Blue", "Summer", 2024, "Casual", "SYNTHETIC missing image Top"])
    pid += 1; (OUT / "images" / f"{pid}.jpg").write_bytes(b"not really a jpeg")
    rows.append([pid, "Women", "Apparel", "Topwear", "Tops", "Blue", "Summer", 2024, "Casual", "SYNTHETIC broken image Top"])
    pid += 1; shutil.copy(OUT / "images" / "900001.jpg", OUT / "images" / f"{pid}.jpg")  # duplicate image, new id
    rows.append([pid, "Women", "Apparel", "Topwear", "Kurtas", "Red", "Summer", 2024, "Ethnic", "SYNTHETIC duplicate image"])
    rows.append(list(rows[0]))  # duplicate product id
    rows.append([pid + 1, "Men", "Apparel", "Topwear", "Kurtas", "Blue", "Summer", 2024, "Ethnic", "SYNTHETIC Men Kurta"])

    with open(OUT / "styles.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "gender", "masterCategory", "subCategory", "articleType", "baseColour", "season",
                     "year", "usage", "productDisplayName"])
        wr.writerows(rows)
    (OUT / "SOURCE.txt").write_text("SYNTHETIC_TEST_FIXTURE generated by scripts/make_test_fixtures.py. "
                                     "Test-only data, not real products.\n")
    print(f"Wrote {len(rows)} synthetic rows to {OUT}")


if __name__ == "__main__":
    main()
