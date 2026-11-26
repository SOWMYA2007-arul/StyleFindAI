"""SYNTHETIC segmentation benchmark set with exact ground-truth masks (regression testing only).

    python scripts/make_segmentation_fixture.py --per-category 20

Writes dataset/evaluation/segmentation_synthetic/ in the "mask folder" format read by
scripts/evaluate_segmentation.py:
    images/<id>.jpg             input photo
    masks/<id>.png              union of all garment pixels that are VISIBLE (255) - ground truth
    instances/<id>__<k>__<label>.png   one mask per garment instance (multi-garment scenes)
    meta.csv                    id, category, n_instances, notes
and an equivalent COCO-style annotations.json (polygons) to test the COCO loader.

These are drawings, not photos. Results on them are regression checks, never real-world accuracy.
Real evaluation: a licensed, annotated dataset (e.g. Fashionpedia, CC BY 4.0 annotations) - see docs/SEGMENTATION.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from _common import DATASET_DIR

OUT = DATASET_DIR / "evaluation" / "segmentation_synthetic"
W, H = 360, 480
CATEGORIES = ["product_only", "white_on_white", "dark_on_dark", "patterned", "gradient_bg", "shadow",
              "complex_bg", "clutter", "thin_details", "transparent", "person_wearing", "multi_garment", "occluded"]
SKIN = [(224, 172, 140), (198, 134, 100), (160, 105, 75), (241, 194, 165), (120, 80, 55)]


def dress_poly(cx, top, bottom, half_top, half_bottom, sleeve):
    pts = [(cx - half_top * 0.4, top), (cx - half_top, top + 12)]
    if sleeve:
        pts += [(cx - half_top - 40, top + 12 + sleeve), (cx - half_top - 22, top + 22 + sleeve)]
    pts += [(cx - half_top * 0.8, top + 90), (cx - half_bottom, bottom), (cx + half_bottom, bottom), (cx + half_top * 0.8, top + 90)]
    if sleeve:
        pts += [(cx + half_top + 22, top + 22 + sleeve), (cx + half_top + 40, top + 12 + sleeve)]
    pts += [(cx + half_top, top + 12), (cx + half_top * 0.4, top)]
    return pts


def texture(rng, base, kind="print"):
    tex = Image.new("RGB", (W, H), base)
    d = ImageDraw.Draw(tex)
    acc = tuple(255 - c for c in base)
    if kind == "stripes":
        for y in range(0, H, 16):
            d.rectangle((0, y, W, y + 5), fill=acc)
    elif kind == "print":
        for _ in range(140):
            x, y, s = rng.randint(0, W), rng.randint(0, H), rng.randint(4, 10)
            d.ellipse((x, y, x + s, y + s), fill=acc)
    return tex


def background(rng, kind):
    if kind == "gradient":
        a = np.linspace(rng.randint(40, 90), rng.randint(180, 230), H)[:, None, None] * np.ones((H, W, 3))
        return Image.fromarray(a.astype(np.uint8))
    if kind == "complex":
        bg = Image.new("RGB", (W, H), tuple(rng.randint(80, 170) for _ in range(3)))
        d = ImageDraw.Draw(bg)
        for _ in range(25):
            x, y = rng.randint(-40, W), rng.randint(-40, H)
            d.rectangle((x, y, x + rng.randint(20, 140), y + rng.randint(20, 140)), fill=tuple(rng.randint(40, 220) for _ in range(3)))
        return bg.filter(ImageFilter.GaussianBlur(1.5))
    if kind == "dark":
        return Image.new("RGB", (W, H), (18, 18, 22))
    if kind == "white":
        return Image.new("RGB", (W, H), (247, 247, 245))
    return Image.new("RGB", (W, H), tuple(rng.randint(225, 250) for _ in range(3)))


def paste(canvas: Image.Image, layer: Image.Image, mask: Image.Image, alpha: float = 1.0):
    m = mask if alpha >= 1 else mask.point(lambda v: int(v * alpha))
    canvas.paste(layer, (0, 0), m)


def poly_mask(pts) -> Image.Image:
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return m


def scene(cat: str, rng: random.Random):
    """Returns (image, [(label, instance_mask_L)], notes)."""
    cx = W / 2 + rng.randint(-20, 20)
    base = tuple(rng.randint(30, 220) for _ in range(3))
    pts = dress_poly(cx, 60, 430, rng.randint(55, 70), rng.randint(90, 130), rng.choice([0, 40, 80]))
    notes = ""
    if cat in ("product_only", "patterned", "gradient_bg", "shadow", "complex_bg", "clutter", "occluded", "thin_details", "transparent"):
        bgkind = {"gradient_bg": "gradient", "complex_bg": "complex"}.get(cat, "plain")
        img = background(rng, bgkind)
        gm = poly_mask(pts)
        if cat == "shadow":
            sh = poly_mask([(x + 28, y + 14) for x, y in pts])
            paste(img, Image.new("RGB", (W, H), tuple(max(0, c - 60) for c in img.getpixel((5, 5)))), sh)
        tex = texture(rng, base, "stripes" if cat == "patterned" and rng.random() < 0.5 else "print") if cat == "patterned" else Image.new("RGB", (W, H), base)
        paste(img, tex, gm)
        if cat == "thin_details":
            # thin straps above the bodice and fringe/tassels along the hem (3-px details)
            d = ImageDraw.Draw(img); dm = ImageDraw.Draw(gm)
            for sx in (cx - 30, cx + 30):
                d.line([(sx, 18), (sx, 64)], fill=base, width=3); dm.line([(sx, 18), (sx, 64)], fill=255, width=3)
            for x in range(int(cx - 110), int(cx + 110), 9):
                d.line([(x, 428), (x + 2, 462)], fill=base, width=2); dm.line([(x, 428), (x + 2, 462)], fill=255, width=2)
            notes = "3px straps + 2px tassels"
        if cat == "transparent":
            # semi-transparent dupatta draped over the garment and background; GT counts it as garment
            dpts = [(cx - 150, 120), (cx - 110, 100), (cx + 160, 330), (cx + 120, 350)]
            dm_ = poly_mask(dpts)
            paste(img, Image.new("RGB", (W, H), (230, 60, 120)), dm_, alpha=0.45)
            gm = Image.fromarray(np.maximum(np.asarray(gm), np.asarray(dm_)))
            notes = "45% opaque dupatta"
        if cat == "clutter":
            d = ImageDraw.Draw(img)
            for _ in range(3):
                x, y = rng.choice([(10, 20), (W - 70, 380), (W - 60, 40), (12, 400)])
                d.ellipse((x, y, x + 50, y + 50), fill=tuple(rng.randint(0, 255) for _ in range(3)))
        if cat == "occluded":
            # a hand/bag in FRONT of the garment: those pixels are not visible garment
            d = ImageDraw.Draw(img)
            ox, oy = int(cx - 60 + rng.randint(-20, 20)), rng.randint(180, 280)
            occ = Image.new("L", (W, H), 0)
            ImageDraw.Draw(occ).rounded_rectangle((ox, oy, ox + 110, oy + 80), radius=25, fill=255)
            paste(img, Image.new("RGB", (W, H), rng.choice(SKIN)), occ)
            gm = Image.fromarray(np.where(np.asarray(occ) > 0, 0, np.asarray(gm)).astype(np.uint8))
            notes = "occluder in front"
        return img.filter(ImageFilter.GaussianBlur(0.7)), [("dress", gm)], notes
    if cat == "white_on_white":
        img = background(rng, "white"); gm = poly_mask(pts)
        paste(img, Image.new("RGB", (W, H), (237, 237, 235)), gm)
        ImageDraw.Draw(img).line(pts + [pts[0]], fill=(214, 214, 212), width=2)
        return img.filter(ImageFilter.GaussianBlur(0.7)), [("dress", gm)], ""
    if cat == "dark_on_dark":
        img = background(rng, "dark"); gm = poly_mask(pts)
        paste(img, Image.new("RGB", (W, H), (34, 30, 38)), gm)
        return img.filter(ImageFilter.GaussianBlur(0.7)), [("dress", gm)], "black garment on near-black"
    if cat in ("person_wearing", "multi_garment"):
        img = background(rng, rng.choice(["plain", "gradient", "complex"]))
        skin = rng.choice(SKIN)
        d = ImageDraw.Draw(img)
        # body drawn first (skin), clothing on top; head + hair, arms, legs remain visible skin
        d.ellipse((cx - 28, 10, cx + 28, 70), fill=skin)                       # face
        d.chord((cx - 31, 4, cx + 31, 50), 180, 360, fill=(30, 22, 20))         # hair
        d.rectangle((cx - 10, 66, cx + 10, 84), fill=skin)                      # neck
        arm = rng.choice(["down", "raised", "out"])
        for side in (-1, 1):
            if arm == "down":
                d.line([(cx + side * 62, 95), (cx + side * 88, 250)], fill=skin, width=20)
            elif arm == "raised":
                d.line([(cx + side * 62, 95), (cx + side * 95, 10)], fill=skin, width=20)
            else:
                d.line([(cx + side * 62, 95), (cx + side * 170, 120)], fill=skin, width=20)
        insts = []
        if cat == "person_wearing":
            top_pts = dress_poly(cx, 80, 400, 62, rng.randint(80, 110), rng.choice([0, 40]))
            gm = poly_mask(top_pts)
            paste(img, Image.new("RGB", (W, H), base), gm)
            d.line([(cx - 25, 400), (cx - 30, 475)], fill=skin, width=18); d.line([(cx + 25, 400), (cx + 30, 475)], fill=skin, width=18)
            insts.append(("dress", gm))
        else:
            # kurti (top) + trousers (bottom) + dupatta: three separate instances
            kpts = dress_poly(cx, 80, 300, 62, 80, 40)
            km = poly_mask(kpts)
            tm = poly_mask([(cx - 60, 296), (cx + 60, 296), (cx + 58, 475), (cx + 8, 475), (cx, 330), (cx - 8, 475), (cx - 58, 475)])
            tm = Image.fromarray(np.where(np.asarray(km) > 0, 0, np.asarray(tm)).astype(np.uint8))
            col_t = tuple(int(255 - c * 0.7) for c in base)
            paste(img, Image.new("RGB", (W, H), col_t), tm)
            paste(img, Image.new("RGB", (W, H), base), km)
            dpts = [(cx - 70, 85), (cx - 45, 80), (cx + 75, 290), (cx + 50, 300)]
            dmask = poly_mask(dpts)
            col_d = (230, 200, 60) if sum(base) < 400 else (40, 80, 160)
            paste(img, Image.new("RGB", (W, H), col_d), dmask)
            dm = np.asarray(dmask) > 0
            insts = [("kurti", Image.fromarray(np.where(dm, 0, np.asarray(km)).astype(np.uint8))),
                     ("trousers", Image.fromarray(np.where(dm, 0, np.asarray(tm)).astype(np.uint8))),
                     ("dupatta", dmask)]
        return img.filter(ImageFilter.GaussianBlur(0.7)), insts, f"arms {arm}"
    raise ValueError(cat)


def polygons(mask: np.ndarray) -> list[list[float]]:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1).astype(float).tolist() for c in cnts if len(c) >= 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)  # only this generated fixture folder
    for sub in ("images", "masks", "instances"):
        (OUT / sub).mkdir(parents=True)
    rng = random.Random(args.seed)
    meta, coco_imgs, coco_anns = [], [], []
    labels = {}
    for cat in CATEGORIES:
        for i in range(args.per_category):
            sid = f"{cat}_{i:03d}"
            img, insts, notes = scene(cat, rng)
            img.save(OUT / "images" / f"{sid}.jpg", quality=92)
            union = np.zeros((H, W), bool)
            iid = len(coco_imgs) + 1
            coco_imgs.append({"id": iid, "file_name": f"{sid}.jpg", "width": W, "height": H, "scene": cat})
            for k, (lab, m) in enumerate(insts):
                mm = np.asarray(m) > 127
                union |= mm
                Image.fromarray((mm * 255).astype(np.uint8)).save(OUT / "instances" / f"{sid}__{k}__{lab}.png")
                cid = labels.setdefault(lab, len(labels) + 1)
                ys, xs = np.where(mm)
                coco_anns.append({"id": len(coco_anns) + 1, "image_id": iid, "category_id": cid, "iscrowd": 0,
                                  "segmentation": polygons(mm), "area": int(mm.sum()),
                                  "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]})
            Image.fromarray((union * 255).astype(np.uint8)).save(OUT / "masks" / f"{sid}.png")
            meta.append({"id": sid, "category": cat, "n_instances": len(insts), "notes": notes})
    with open(OUT / "meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0])); w.writeheader(); w.writerows(meta)
    (OUT / "annotations.json").write_text(json.dumps({
        "info": {"description": "SYNTHETIC StyleFind segmentation fixture - not real photos"},
        "images": coco_imgs, "annotations": coco_anns,
        "categories": [{"id": v, "name": k} for k, v in labels.items()]}))
    (OUT / "SOURCE.txt").write_text("Generated by scripts/make_segmentation_fixture.py. Synthetic drawings, test-only.\n")
    print(f"wrote {len(meta)} synthetic scenes ({len(CATEGORIES)} categories) to {OUT}")


if __name__ == "__main__":
    main()
