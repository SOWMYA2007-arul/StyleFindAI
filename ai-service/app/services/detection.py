"""Garment isolation for searches with crop=true.

Modes (GARMENT_ISOLATION in .env):
  off         - always the full image
  auto        - (default) segmentation pipeline (app/services/segmentation): routes each photo to the
                cheapest adequate method; uses whatever optional models .env enables (clothing YOLO,
                SegFormer human parser, SAM 2, MediaPipe pose). With none enabled it is weight-free.
  foreground  - the same pipeline restricted to weight-free methods (strategy "light")
  yolo        - pipeline with the YOLO garment detector switched on (YOLO_WEIGHTS)

Honest caveats:
* Standard Ultralytics weights are COCO-trained: 'person' exists, garment classes do not. COCO
  weights cannot detect garments; they can only provide person boxes/masks (PERSON_SEGMENTER=yolo).
  For garment boxes point YOLO_WEIGHTS at a clothing model (e.g. trained on DeepFashion2) and check
  the licence of those weights.
* Without a parser / clothing detector, skin removal on person photos is colour-based and can be
  wrong for garments close to the wearer's skin tone; the result then carries a low confidence and
  a note, and below MIN_SEGMENTATION_CONFIDENCE the full image is used.

Every failure falls back (models -> weight-free -> full image) - search never breaks here.
The v2 functions (foreground_isolation, detect_box, sam_mask) are kept for scripts and comparison.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.config import get_settings
from app.services.foreground import estimate_foreground
from app.utils.logging import get_logger

log = get_logger("detection")

GARMENT_WORDS = ("dress", "skirt", "top", "shirt", "sleeve", "outwear", "trousers", "shorts", "sling",
                 "vest", "kurta", "saree", "lehenga", "garment", "clothing", "jacket", "coat")
_yolo = None
_yolo_failed = False
_sam = None


def _get_yolo():
    global _yolo, _yolo_failed
    if _yolo is None and not _yolo_failed:
        try:
            from ultralytics import YOLO  # optional dependency (AGPL-3.0)

            _yolo = YOLO(get_settings().yolo_weights)
        except Exception as e:  # noqa: BLE001 - missing package / weights / download failure
            _yolo_failed = True
            log.warning("YOLO unavailable (%s); using foreground isolation", e)
    return _yolo


def detect_box(img: Image.Image) -> tuple[tuple[int, int, int, int] | None, str, float]:
    model = _get_yolo()
    if model is None:
        return None, "", 0.0
    s = get_settings()
    res = model.predict(np.asarray(img), verbose=False, conf=s.detection_min_confidence)[0]
    names = res.names
    best, best_score, label, conf = None, 0.0, "", 0.0
    for b in res.boxes:
        cls_name = names[int(b.cls)].lower()
        garment = any(w in cls_name for w in GARMENT_WORDS)
        if not garment and cls_name != "person":
            continue
        x0, y0, x1, y1 = [int(v) for v in b.xyxy[0].tolist()]
        c = float(b.conf)
        score = (x1 - x0) * (y1 - y0) * c * (2.0 if garment else 1.0)   # prefer real garment classes
        if score > best_score:
            best, best_score, label, conf = (x0, y0, x1, y1), score, cls_name, c
    return best, label, conf


def sam_mask(img: Image.Image, box) -> np.ndarray | None:
    global _sam
    s = get_settings()
    if not (s.enable_sam2 and s.sam2_checkpoint):
        return None
    from sam2.build_sam import build_sam2  # optional dependency (Apache-2.0)
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if _sam is None:
        _sam = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", s.sam2_checkpoint, device=s.ai_device))
    _sam.set_image(np.asarray(img))
    masks, _, _ = _sam.predict(box=np.array(box)[None, :], multimask_output=False)
    return masks[0].astype(bool)


def _padded(box, w: int, h: int, pad: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)


def _whiten(img: Image.Image, mask_full: np.ndarray) -> Image.Image:
    arr = np.asarray(img).copy()
    arr[~mask_full] = 255
    return Image.fromarray(arr)


def foreground_isolation(img: Image.Image) -> tuple[Image.Image, dict]:
    s = get_settings()
    fg = estimate_foreground(img, refine=True)
    if fg.method == "centre_ellipse" or fg.confidence < 0.4:
        return img, {"method": "full_image", "reason": "garment not separable from background",
                     "mask_method": fg.method}
    w, h = img.size
    mask_full = np.asarray(Image.fromarray(fg.mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127
    ys, xs = np.where(mask_full)
    if len(ys) == 0 or mask_full.mean() > 0.92:
        return img, {"method": "full_image", "reason": "garment already fills the photo", "mask_method": fg.method}
    box = _padded((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), w, h, s.isolation_padding)
    # dilate the mask slightly so fringes, tassels and thin sleeves at the edge are not cut away
    import cv2

    k = max(3, int(0.015 * max(w, h)) | 1)
    mask_full = cv2.dilate(mask_full.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    out = _whiten(img, mask_full).crop(box)
    return out, {"method": "foreground_crop", "mask_method": fg.method, "mask_confidence": round(fg.confidence, 2),
                 "box": list(box), "garment_share": round(float(fg.mask.mean()), 3)}


def _legacy_name(method: str) -> str:
    base = method.split("+")[0]
    return {"foreground": "foreground_crop", "foreground_split": "foreground_crop",
            "garment_detector": "yolo_crop"}.get(base, f"{base}_crop")


def isolate_garment(img: Image.Image, image_key: str | None = None, target=None) -> tuple[Image.Image, dict]:
    """Return (image_for_encoder, info). info['method'] tells the UI what happened (v2-compatible names:
    foreground_crop / yolo_crop / full_image, plus classical_person_crop, human_parser_crop ...).
    info['segmentation'] carries the full structured result (path, instances, timings, fallbacks)."""
    from app.services.segmentation.pipeline import isolate

    s = get_settings()
    mode = (s.garment_isolation or "auto").lower()
    if s.enable_garment_detection and mode == "foreground":
        mode = "yolo"  # backward compatibility with ENABLE_GARMENT_DETECTION=true
    if mode == "off":
        return img, {"method": "full_image", "reason": "garment isolation disabled"}
    strategy = "light" if mode == "foreground" else None
    try:
        out, seg = isolate(img, image_key, target, strategy)
    except Exception as e:  # noqa: BLE001 - never break search
        log.warning("segmentation failed (%s); trying v2 foreground isolation", e)
        out, info = foreground_isolation(img)
        info["fallback_from"] = f"segmentation pipeline error: {type(e).__name__}"
        return out, info
    tg = (seg.get("instances") or [None])[seg["target_index"]] if seg.get("target_index") is not None else None
    if seg.get("fallback_used") or tg is None:
        info = {"method": "full_image", "reason": seg.get("fallback_reason") or "no garment isolated"}
    else:
        info = {"method": _legacy_name(seg["method"]), "label": tg.get("label"), "confidence": seg.get("confidence"),
                "box": seg.get("crop_box") or tg.get("bbox")}
    if mode == "yolo" and not str(seg.get("method", "")).startswith("garment_detector"):
        why = next((a.get("reason") for a in seg.get("attempts", []) if a.get("method") == "garment_detector"), None)
        info["fallback_from"] = f"yolo ({why or 'no confident garment detection'})"
    info["segmentation"] = seg
    return out, info
