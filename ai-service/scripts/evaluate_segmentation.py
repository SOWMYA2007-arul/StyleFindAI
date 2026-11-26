"""Garment segmentation benchmark (real or synthetic annotated data).

Datasets
  --format maskdir  --root DIR      DIR/images/*.jpg|png, DIR/masks/<stem>.png (garment = white),
                                    optional DIR/instances/<stem>__<k>__<label>.png, optional DIR/meta.csv
                                    (id,category,...) for per-category results
  --format coco     --images DIR --annotations FILE.json [--garment-categories "dress,skirt,..."]
                                    COCO polygons (RLE too if pycocotools is installed). Works with
                                    Fashionpedia (CC BY 4.0 annotations) and DeepFashion2 converted to COCO.
                                    Per-category grouping: image field "scene" or --meta CSV (file_name,category).

Methods (--methods, default all that are available)
  legacy_v1         original v1 foreground (single border colour)
  foreground        current weight-free foreground mask, plain upsampling (what v2 used)
  foreground_guided same + guided-filter boundary refinement
  pipeline_light    full pipeline, weight-free methods only
  pipeline_auto     full pipeline with whatever optional models .env enables (YOLO / parser / SAM 2 / pose)

Metrics per image: IoU, Dice, precision, recall, boundary F1 (tol = 0.8 % of the diagonal and 2 px),
latency; per method: CPU RSS peak (psutil), GPU peak memory (CUDA only). With instance ground truth:
mean best-match instance IoU and class-agnostic AP50 / mAP@[.5:.95]. A pipeline fallback to the full
image is scored as a full-image mask (that is what the search model would see).

    python scripts/evaluate_segmentation.py --format maskdir --root ..\\dataset\\evaluation\\segmentation_synthetic
    python scripts/evaluate_segmentation.py --format coco --images D:\\fashionpedia\\val2020 --annotations D:\\fashionpedia\\instances_attributes_val2020.json --limit 1000
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from _common import DATASET_DIR, PROJECT_ROOT
from app.config import get_settings
from app.services.foreground import estimate_foreground, legacy_foreground_mask
from app.services.preprocess import to_rgb
from app.services.segmentation.pipeline import segment
from app.services.segmentation.refine import boundary_f1, dice, iou, precision_recall, refine_to_image

FASHIONPEDIA_GARMENTS = ("shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest", "pants",
                         "shorts", "skirt", "coat", "dress", "jumpsuit", "cape")


# ------------------------------------------------------------------ loaders
def load_maskdir(root: Path):
    meta = {}
    if (root / "meta.csv").exists():
        with open(root / "meta.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                meta[r.get("id") or Path(r.get("file_name", "")).stem] = r.get("category", "all")
    for p in sorted((root / "images").iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        mp = root / "masks" / f"{p.stem}.png"
        if not mp.exists():
            continue
        insts = []
        for ip in sorted((root / "instances").glob(f"{p.stem}__*.png")) if (root / "instances").exists() else []:
            insts.append((ip.stem.split("__")[-1], ip))
        yield {"id": p.stem, "image": p, "mask": mp, "instances": insts, "category": meta.get(p.stem, "all")}


def _poly_mask(seg, h, w):
    m = np.zeros((h, w), np.uint8)
    if isinstance(seg, list):
        for poly in seg:
            pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(m, [pts], 1)
        return m > 0
    try:
        from pycocotools import mask as mask_utils  # optional, for RLE

        rle = seg if isinstance(seg.get("counts"), (bytes, str)) else mask_utils.frPyObjects(seg, h, w)
        return mask_utils.decode(rle).astype(bool)
    except ImportError:
        raise SystemExit("RLE segmentation found; install pycocotools (pip install pycocotools)") from None


def load_coco(images: Path, ann: Path, garment_names: list[str] | None, meta_csv: Path | None):
    data = json.loads(Path(ann).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    names = set(n.strip().lower() for n in garment_names) if garment_names else None
    if names is None and set(FASHIONPEDIA_GARMENTS) <= {v.lower() for v in cats.values()}:
        names = set(FASHIONPEDIA_GARMENTS)  # Fashionpedia: ignore garment PARTS (sleeve, neckline, pocket...)
    keep = {cid for cid, n in cats.items() if names is None or n.lower() in names}
    meta = {}
    if meta_csv:
        with open(meta_csv, newline="", encoding="utf-8") as f:
            meta = {r["file_name"]: r.get("category", "all") for r in csv.DictReader(f)}
    by_img = defaultdict(list)
    for a in data["annotations"]:
        if a["category_id"] in keep and not a.get("iscrowd"):
            by_img[a["image_id"]].append(a)
    for im in data["images"]:
        anns = by_img.get(im["id"])
        if not anns:
            continue
        yield {"id": Path(im["file_name"]).stem, "image": images / im["file_name"], "coco": (im, anns, cats),
               "category": im.get("scene") or meta.get(im["file_name"], "all")}


def gt_masks(item, size):
    w, h = size
    if "coco" in item:
        im, anns, cats = item["coco"]
        sx, sy = w / im["width"], h / im["height"]
        insts = []
        for a in anns:
            m = _poly_mask(a["segmentation"], im["height"], im["width"])
            if (sx, sy) != (1, 1):
                m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127
            insts.append((cats[a["category_id"]], m))
    else:
        insts = [(lab, np.asarray(Image.open(p).convert("L").resize((w, h), Image.Resampling.NEAREST)) > 127)
                 for lab, p in item["instances"]]
        union = np.asarray(Image.open(item["mask"]).convert("L").resize((w, h), Image.Resampling.NEAREST)) > 127
        return union, insts
    union = np.zeros((h, w), bool)
    for _, m in insts:
        union |= m
    return union, insts


# ------------------------------------------------------------------ methods
def _up(mask, size):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)) > 127


def run_method(name: str, img: Image.Image):
    """Returns (union_mask, [(confidence, mask)], info)."""
    w, h = img.size
    if name == "legacy_v1":
        _, m = legacy_foreground_mask(img, 320)
        m = _up(m, (w, h))
        return m, [(0.5, m)], {}
    if name in ("foreground", "foreground_guided"):
        fg = estimate_foreground(img, 320)
        if name == "foreground":
            m = _up(fg.mask, (w, h))
        else:
            m, _ = refine_to_image(fg.mask, np.asarray(img))
        return m, [(fg.confidence, m)], {"method": fg.method}
    strategy = "light" if name == "pipeline_light" else "auto"
    r = segment(img, None, None, strategy)
    insts = [(g.confidence, g.mask) for g in r.instances if g.mask is not None]
    if r.fallback_used or r.target_index is None:
        union = np.ones((h, w), bool)  # the search model sees the full image
    else:
        union = np.zeros((h, w), bool)
        for _, m in insts:
            union |= m
    return union, insts, {"path": r.path, "method": r.method, "fallback": r.fallback_used, "confidence": r.confidence}


def ap_scores(preds: list[tuple[float, np.ndarray]], gts: list[np.ndarray], thresholds) -> dict:
    """Class-agnostic AP per IoU threshold for ONE image's predictions (accumulated later)."""
    out = {}
    for t in thresholds:
        matched, tps = set(), []
        for conf, pm in sorted(preds, key=lambda x: -x[0]):
            best, bj = 0.0, -1
            for j, gm in enumerate(gts):
                if j in matched:
                    continue
                v = iou(pm, gm)
                if v > best:
                    best, bj = v, j
            ok = best >= t
            if ok:
                matched.add(bj)
            tps.append((conf, ok))
        out[t] = tps
    return out


def average_precision(tps: list[tuple[float, bool]], n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    tps = sorted(tps, key=lambda x: -x[0])
    tp = np.cumsum([1 if ok else 0 for _, ok in tps]); fp = np.cumsum([0 if ok else 1 for _, ok in tps])
    if len(tps) == 0:
        return 0.0
    rec, prec = tp / n_gt, tp / np.maximum(1, tp + fp)
    ap = 0.0
    for r in np.linspace(0, 1, 101):  # COCO-style 101-point interpolation
        p = prec[rec >= r]
        ap += (p.max() if len(p) else 0.0) / 101
    return float(ap)


def rss_mb():
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=["maskdir", "coco"], default="maskdir")
    ap.add_argument("--root", type=Path, default=DATASET_DIR / "evaluation" / "segmentation_synthetic")
    ap.add_argument("--images", type=Path); ap.add_argument("--annotations", type=Path)
    ap.add_argument("--garment-categories", default=None, help="comma list of category names counted as garments")
    ap.add_argument("--meta", type=Path, help="CSV file_name,category for per-category results (coco)")
    ap.add_argument("--methods", nargs="+", default=["legacy_v1", "foreground", "foreground_guided", "pipeline_light", "pipeline_auto"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=800, help="resize large photos (like the search service does)")
    ap.add_argument("--tag", default="seg")
    ap.add_argument("--synthetic", action="store_true", help="mark results as synthetic (auto for the fixture)")
    args = ap.parse_args()

    if args.format == "maskdir":
        items = list(load_maskdir(args.root))
        synthetic = args.synthetic or (args.root / "SOURCE.txt").exists()
        try:
            src = str(args.root.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            src = str(args.root)
    else:
        items = list(load_coco(args.images, args.annotations, args.garment_categories.split(",") if args.garment_categories else None, args.meta))
        synthetic, src = args.synthetic, f"{args.annotations}"
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} images, methods {args.methods}")
    try:
        import torch

        cuda = torch.cuda.is_available() and get_settings().ai_device == "cuda"
    except Exception:  # noqa: BLE001
        cuda = False
    rows, ap_acc, n_gt_inst, mem = [], defaultdict(lambda: defaultdict(list)), defaultdict(int), {}
    for name in args.methods:
        peak = rss_mb() or 0
        if cuda:
            import torch

            torch.cuda.reset_peak_memory_stats()
        for k, it in enumerate(items):
            with Image.open(it["image"]) as im:
                img = to_rgb(ImageOps.exif_transpose(im))
            if max(img.size) > args.max_side:
                img.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
            gt, gt_inst = gt_masks(it, img.size)
            t = time.perf_counter()
            try:
                pred, insts, info = run_method(name, img)
            except Exception as e:  # noqa: BLE001
                pred, insts, info = np.ones_like(gt), [], {"error": f"{type(e).__name__}: {e}"[:200]}
            ms = (time.perf_counter() - t) * 1000
            p, r = precision_recall(pred, gt)
            row = {"method": name, "id": it["id"], "category": it["category"], "iou": iou(pred, gt), "dice": dice(pred, gt),
                   "precision": p, "recall": r, "bf1": boundary_f1(pred, gt), "bf1_2px": boundary_f1(pred, gt, 2),
                   "ms": ms, **{f"info_{a}": b for a, b in info.items()}}
            if len(gt_inst) > 1 or (gt_inst and name.startswith("pipeline")):
                gms = [m for _, m in gt_inst]
                row["inst_iou"] = float(np.mean([max([iou(pm, gm) for _, pm in insts] or [0.0]) for gm in gms]))
                for th, tps in ap_scores(insts, gms, [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]).items():
                    ap_acc[name][th] += tps
                n_gt_inst[name] += len(gms)
            rows.append(row)
            peak = max(peak, rss_mb() or 0)
            if (k + 1) % 100 == 0:
                print(f"  {name}: {k + 1}/{len(items)}", flush=True)
        mem[name] = {"cpu_rss_peak_mb": round(peak, 1) if peak else None}
        if cuda:
            import torch

            mem[name]["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)

    def agg(rs):
        keys = ["iou", "dice", "precision", "recall", "bf1", "bf1_2px"]
        out = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        out["ms_p50"] = float(np.percentile([r["ms"] for r in rs], 50)); out["ms_p95"] = float(np.percentile([r["ms"] for r in rs], 95))
        ii = [r["inst_iou"] for r in rs if "inst_iou" in r]
        out["inst_iou"] = float(np.mean(ii)) if ii else None
        fb = [r for r in rs if "info_fallback" in r]
        out["fallback_rate"] = float(np.mean([r["info_fallback"] for r in fb])) if fb else None
        out["n"] = len(rs)
        return out

    summary = {}
    for name in args.methods:
        rs = [r for r in rows if r["method"] == name]
        cats = sorted({r["category"] for r in rs})
        aps = {th: average_precision(tps, n_gt_inst[name]) for th, tps in ap_acc[name].items()}
        summary[name] = {"overall": agg(rs), "by_category": {c: agg([r for r in rs if r["category"] == c]) for c in cats},
                         "AP50": aps.get(0.5), "mAP": float(np.nanmean(list(aps.values()))) if aps else None, **mem[name]}
    stamp = datetime.now(timezone.utc)
    meta = {"generated": stamp.strftime("%Y-%m-%d %H:%M UTC"), "source": src, "format": args.format, "n_images": len(items),
            "synthetic": synthetic, "max_side": args.max_side,
            "settings": {k: getattr(get_settings(), k) for k in ("segmentation_strategy", "garment_detector", "person_segmenter",
                                                                    "human_parser", "enable_sam2", "pose_estimator",
                                                                    "boundary_refinement", "min_segmentation_confidence", "ai_device")}}
    out_dir = DATASET_DIR / "evaluation" / "experiments" / f"{stamp:%Y%m%d-%H%M%S}_segmentation_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "result.json").write_text(json.dumps({"meta": meta, "summary": summary, "per_image": rows}, indent=2, default=float))
    fb_txt = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
    pct = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    L = [f"# Segmentation benchmark ({args.tag})", "", f"Generated {meta['generated']} by `scripts/evaluate_segmentation.py`.",
         f"Data: `{src}` ({len(items)} images). Settings: {meta['settings']}", ""]
    if synthetic:
        L += ["> **SYNTHETIC drawings** with exact masks: regression evidence only, NOT real-world accuracy.", ""]
    L += ["## Overall", "", "| method | IoU | Dice | precision | recall | boundary F1 | BF1 (2 px) | inst. IoU | AP50 | mAP | fallback | p50 ms | p95 ms | CPU RSS peak MB |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, sm in summary.items():
        o = sm["overall"]
        L.append(f"| {name} | {o['iou']:.3f} | {o['dice']:.3f} | {o['precision']:.3f} | {o['recall']:.3f} | {o['bf1']:.3f} | "
                 f"{o['bf1_2px']:.3f} | {pct(o['inst_iou'])} | {pct(sm['AP50'])} | {pct(sm['mAP'])} | "
                 f"{fb_txt(o['fallback_rate'])} | "
                 f"{o['ms_p50']:.0f} | {o['ms_p95']:.0f} | {sm.get('cpu_rss_peak_mb')} |")
    cats = sorted({r["category"] for r in rows})
    for metric, label in (("iou", "IoU"), ("bf1", "boundary F1")):
        L += ["", f"## {label} by image category", "", "| category | " + " | ".join(args.methods) + " |",
              "|---|" + "---|" * len(args.methods)]
        for c in cats:
            L.append(f"| {c} | " + " | ".join(f"{summary[m]['by_category'][c][metric]:.3f}" for m in args.methods
                                             if c in summary[m]["by_category"]) + " |")
    L += ["", "Notes: a pipeline fallback is scored as the full image. boundary F1 tolerance = 0.8 % of the image diagonal;",
          "BF1 (2 px) is the strict version that rewards thin details (straps, tassels, lace edges)."]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    (PROJECT_ROOT / "docs" / "SEGMENTATION_BENCHMARK.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12 + len(summary)]))
    print(f"\nwrote {out_dir} and docs/SEGMENTATION_BENCHMARK.md")


if __name__ == "__main__":
    main()
