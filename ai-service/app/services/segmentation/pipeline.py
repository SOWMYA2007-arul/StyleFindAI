"""Garment segmentation pipeline: choose the cheapest adequate method per photo.

  quality + foreground + scene analysis (weight-free, ~15-30 ms)
      |
      +-- product_only : foreground mask (+ boundary refinement). No YOLO / parser / SAM 2.
      +-- person       : human parser -> clothing detector -> weight-free skin/face/hair removal
      +-- multi_garment: clothing detector (instances) -> colour/connectivity split
      +-- difficult    : detector / parser if configured -> GrabCut-refined foreground
      |
      [optional] pose (person path only), SAM 2 refinement of the TARGET garment (only when its
                 confidence is not already high)
      |
      confidence check -> below MIN_SEGMENTATION_CONFIDENCE the result falls back to the full image
                          (reason recorded) instead of returning a doubtful mask.

Every attempt (method, ok, reason, ms) and every stage time is recorded. Results are cached by
(image hash, configuration, target).
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image

from app.config import get_settings
from app.services.foreground import estimate_foreground
from app.services.segmentation import backends as B
from app.services.segmentation.cache import LRUCache
from app.services.segmentation.classical import (analyze_scene, person_clothing, product_instances, remove_occluders,
                                                 to_image)
from app.services.segmentation.quality import assess
from app.services.segmentation.refine import occlusion_ratio, refine_to_image, remove_specks
from app.services.segmentation.types import GarmentInstance, SegmentationResult, bbox_of, valid_bbox
from app.utils.logging import get_logger

log = get_logger("segmentation")
WORK_SIZE = 320
GROUP_PRIORITY = {"full_body": 4, "upper": 3, "outer": 2, "lower": 2, "garment": 2, "accessory": 1}
_cache: LRUCache | None = None


def cache() -> LRUCache:
    global _cache
    if _cache is None:
        _cache = LRUCache(get_settings().segmentation_cache_size)
    return _cache


def config_signature() -> tuple:
    s = get_settings()
    return (s.segmentation_strategy, s.garment_detector, s.yolo_weights, s.person_segmenter, s.person_yolo_weights,
            s.human_parser, s.human_parser_model, s.enable_sam2, s.sam2_checkpoint, s.pose_estimator,
            s.detection_min_confidence, s.detection_min_box_area, s.min_segmentation_confidence, s.boundary_refinement)


class _Clock:
    def __init__(self):
        self.t: dict[str, float] = {}

    def run(self, name, fn, *a, **kw):
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t) * 1000


def _try(res: SegmentationResult, clock: _Clock, method: str, fn, *a, timeout: bool = True, **kw):
    """Run one optional method; record the attempt; never raise."""
    s = get_settings()
    t = time.perf_counter()
    try:
        out = B.run_with_timeout(fn, s.segmentation_timeout_s, *a, **kw) if timeout else fn(*a, **kw)
        ok = out is not None and (not isinstance(out, (list, tuple)) or len(out) > 0)
        res.attempts.append({"method": method, "ok": bool(ok), "reason": None if ok else "no garment found",
                             "ms": round((time.perf_counter() - t) * 1000, 1)})
        return out if ok else None
    except Exception as e:  # noqa: BLE001 - optional model failure must not break search
        res.attempts.append({"method": method, "ok": False, "reason": f"{type(e).__name__}: {str(e)[:120]}",
                             "ms": round((time.perf_counter() - t) * 1000, 1)})
        return None
    finally:
        clock.t[method] = clock.t.get(method, 0.0) + (time.perf_counter() - t) * 1000


def _choose_target(instances: list[GarmentInstance], target) -> int | None:
    if not instances:
        return None
    if isinstance(target, int) or (isinstance(target, str) and target.isdigit()):
        i = int(target)
        return i if 0 <= i < len(instances) else None
    if isinstance(target, str) and target:
        t = target.lower()
        for i, g in enumerate(instances):
            if t == g.group or t in g.label.lower():
                return i
        return None
    return max(range(len(instances)),
               key=lambda i: (GROUP_PRIORITY.get(instances[i].group, 1), instances[i].area * instances[i].confidence))


def _implausible(bbox, size) -> tuple[float, str]:
    """On cluttered photos the colour model sometimes grabs a background patch. A small region jammed
    into a corner, far from the centre, is unlikely to be the garment the user photographed."""
    if not bbox:
        return 1.0, ""
    w, h = size
    x0, y0, x1, y1 = bbox
    touches = (x0 <= 1) + (y0 <= 1) + (x1 >= w - 1) + (y1 >= h - 1)
    share = (x1 - x0) * (y1 - y0) / float(w * h)
    cx, cy = (x0 + x1) / 2 / w - 0.5, (y0 + y1) / 2 / h - 0.5
    if touches >= 2 and share < 0.15 and max(abs(cx), abs(cy)) > 0.25:
        return 0.4, "small region in a corner; probably background"
    return 1.0, ""


def _to_full(inst: GarmentInstance, rgb_full: np.ndarray, scale_from: tuple[int, int] | None) -> GarmentInstance:
    """Bring a work-resolution mask to image resolution (guided refinement keeps thin details)."""
    if inst.mask is None or scale_from is None:
        return inst
    h, w = rgb_full.shape[:2]
    if inst.mask.shape == (h, w):
        return inst
    if get_settings().boundary_refinement == "guided":
        m, a = refine_to_image(inst.mask, rgb_full)
    else:
        m, a = to_image(inst.mask, (w, h)), None
    m = remove_specks(m, min_area_frac=0.0005)
    inst.mask, inst.alpha = m, a
    inst.bbox = bbox_of(m) or inst.bbox
    return inst


def segment(img: Image.Image, image_key: str | None = None, target=None, strategy: str | None = None) -> SegmentationResult:
    s = get_settings()
    strategy = (strategy or s.segmentation_strategy or "auto").lower()
    key = (image_key, strategy, str(target), config_signature()) if image_key else None
    if key is not None:
        hit = cache().get(key)
        if hit is not None:
            hit.cached = True
            return hit
    w, h = img.size
    clock = _Clock()
    t0 = time.perf_counter()
    res = SegmentationResult(path="disabled", method="full_image", confidence=0.0, image_size=(w, h))
    if strategy == "off":
        res.fallback_reason = "segmentation disabled"
        return res

    rgb_full = np.asarray(img.convert("RGB"))
    fg = clock.run("foreground", estimate_foreground, img, WORK_SIZE)
    res.quality = clock.run("quality", assess, img, float(fg.mask.mean()))
    scene = clock.run("scene", analyze_scene, fg.rgb, fg)
    res.scene = {k: v for k, v in scene.items() if not k.startswith("_") and k != "face_boxes"}
    work = fg.mask.shape[::-1]
    if scene["person"]:
        res.path = "person"
    elif scene["components"] >= 2:
        res.path = "multi_garment"
    elif fg.method.startswith("colour") and fg.confidence >= 0.8:
        res.path = "product_only"
    else:
        res.path = "difficult"
    if strategy == "light" and res.path != "product_only":
        res.scene["note"] = "strategy=light: optional models skipped"

    instances: list[GarmentInstance] = []
    from_work = None  # set when masks come at work resolution
    advanced = strategy == "auto" and res.quality.get("advanced_worthwhile", True)
    body_mask = None
    person_work = None  # person mask from the person segmenter (work resolution), reused below

    # The weight-free person cue (face / head blob + skin) misses many real photos. When a person segmenter
    # is enabled, let it confirm people on ambiguous photos (never on confident product-only photos).
    if res.path in ("difficult", "multi_garment") and advanced and (ps := B.get_backend("person")) is not None:
        persons = _try(res, clock, "person_segmenter", ps.instances, img, "person")
        persons = [g for g in (persons or []) if g.mask is not None and g.mask.mean() >= 0.04]
        if persons:
            main = max(persons, key=lambda g: g.area * g.confidence)
            person_work = np.asarray(Image.fromarray(main.mask.astype(np.uint8) * 255).resize(work, Image.Resampling.NEAREST)) > 127
            res.path = "person"
            res.scene.update(person=True, person_source="person_segmenter", persons_detected=len(persons))

    if res.path == "product_only":
        # cheapest path: the foreground mask is the garment
        m = remove_specks(fg.mask)
        m, occl = remove_occluders(fg.rgb, m, scene["_skin"])
        conf = fg.confidence * (1.0 - min(0.5, 1.5 * occl))
        instances = [GarmentInstance("garment", "garment", round(conf, 3), bbox_of(m), m, "foreground", occlusion=round(occl, 3))]
        if occl > 0:
            res.scene["occluded_share"] = round(occl, 3)
        from_work = work
        res.method = "foreground"
    else:
        if advanced and res.path == "person" and (p := B.get_backend("parser")) is not None:
            out = _try(res, clock, "human_parser", p.instances, img)
            if out:
                instances, body_mask = out[0], out[1]
                res.method = "human_parser"
        if not instances and advanced and (d := B.get_backend("garments")) is not None:
            out = _try(res, clock, "garment_detector", d.instances, img, "garments")
            if out:
                instances, res.method = out, "garment_detector"
                for g in instances:
                    if g.mask is not None:
                        g.occlusion = round(occlusion_ratio(g.mask), 3)
        if not instances and res.path == "person":
            person = fg.mask
            if person_work is not None:
                person = person_work
            else:
                persons = None
                if advanced and (ps := B.get_backend("person")) is not None:
                    persons = _try(res, clock, "person_segmenter", ps.instances, img, "person")
                    persons = [g for g in (persons or []) if g.mask is not None]
                if persons:
                    pm = max(persons, key=lambda g: g.area * g.confidence).mask
                    person = np.asarray(Image.fromarray(pm.astype(np.uint8) * 255).resize(work, Image.Resampling.NEAREST)) > 127
                else:
                    person = clock.run("grabcut", estimate_foreground, img, WORK_SIZE, True).mask
            inst, conf, details = clock.run("classical_person", person_clothing, fg.rgb, person, scene)
            res.attempts.append({"method": "classical_person", "ok": bool(inst), "reason": details.get("note"),
                                 "ms": round(clock.t.get("classical_person", 0), 1)})
            res.scene.update(details)
            instances, from_work, res.method = inst, work, "classical_person"
        if not instances:
            fgr = fg if res.path == "multi_garment" else clock.run("grabcut", estimate_foreground, img, WORK_SIZE, True)
            instances = clock.run("classical_split", product_instances, fgr.rgb, fgr)
            for g in instances:
                g.confidence = round(fgr.confidence * (0.85 if len(instances) > 1 else 1.0), 3)
            res.attempts.append({"method": "foreground_split", "ok": bool(instances), "reason": None,
                                 "ms": round(clock.t.get("classical_split", 0), 1)})
            from_work, res.method = work, "foreground_split" if len(instances) > 1 else "foreground"
            if res.path == "difficult" and fgr.method == "centre_ellipse":
                for g in instances:
                    g.confidence = 0.2
            if res.path == "difficult":
                for g in instances:
                    pen, why = _implausible(g.bbox, work)
                    if pen < 1.0:
                        g.confidence = round(g.confidence * pen, 3)
                        res.scene["implausible_region"] = why
            if scene.get("ambiguous_skin") and res.path == "difficult":
                res.scene["note"] = "most of the figure is skin-coloured; if a person is wearing it, a parsing model is needed"

    # pose (person path only): remove exposed forearm skin the colour model missed; hips -> upper/lower
    if res.path == "person" and advanced and (pose := B.get_backend("pose")) is not None and instances:
        lm = _try(res, clock, "pose", pose.landmarks, img)
        if lm:
            full = [_to_full(g, rgb_full, from_work) for g in instances]
            from_work = None
            limbs = B.limb_mask(lm, (h, w), max(6, w // 40))
            skin_full = to_image(scene["_skin"], (w, h))
            hy = B.hip_line(lm)
            for g in full:
                g.mask = g.mask & ~(limbs & skin_full)
                if hy is not None and g.bbox:
                    g.group = "upper" if g.bbox[3] < hy + 0.1 * h else ("lower" if g.bbox[1] > hy - 0.1 * h else g.group)
            instances = [g for g in full if g.mask.any()]

    instances = [_to_full(g, rgb_full, from_work) for g in instances]
    instances = [g for g in instances if g.mask is None or g.mask.any()]
    for g in instances:
        if g.mask is not None:
            g.bbox = bbox_of(g.mask) or g.bbox
    instances = [g for g in instances if valid_bbox(g.bbox, w, h)]
    res.instances = instances
    ti = _choose_target(instances, target)
    if target is not None and ti is None and instances:
        res.fallback_reason = f"target '{target}' not found; using the most prominent garment"
        ti = _choose_target(instances, None)

    # SAM 2: refine only the target, only when it is not already confident (never for product-only)
    if ti is not None and res.path != "product_only" and advanced and (sam := B.get_backend("sam2")) is not None \
            and instances[ti].confidence < 0.75:
        tg = instances[ti]
        exclude = body_mask if body_mask is not None else (to_image(scene["_skin"], (w, h)) if res.path == "person" else None)
        out = _try(res, clock, "sam2", sam.refine, rgb_full, tg.bbox, tg.mask, exclude, image_key)
        if out and out[0] is not None:
            m, info = out
            cons = max(c["consistency"] for c in info["candidates"]) if info["candidates"] else 0
            if tg.mask is None or cons >= 0.5:
                tg.mask, tg.bbox, tg.source = m, bbox_of(m) or tg.bbox, tg.source + "+sam2"
                tg.confidence = round(min(0.95, max(tg.confidence, 0.5 + 0.4 * cons)), 3)
                res.method += "+sam2"
            else:
                res.attempts[-1]["reason"] = "SAM 2 mask disagreed with the garment estimate; kept the estimate"

    res.target_index = ti
    if ti is None:
        res.fallback_used, res.method = True, "full_image"
        res.fallback_reason = res.fallback_reason or "no garment could be isolated"
        res.confidence = 0.0
    else:
        res.confidence = float(instances[ti].confidence)
        if res.confidence < s.min_segmentation_confidence:
            res.fallback_used = True
            res.fallback_reason = (f"segmentation confidence {res.confidence:.2f} below "
                                   f"{s.min_segmentation_confidence:.2f}; using the full image")
            res.method = f"full_image (low-confidence {res.method})"
            res.target_index = None
    res.timings_ms = dict(clock.t, total=(time.perf_counter() - t0) * 1000)
    if key is not None:
        cache().put(key, res)
    return res


def isolate(img: Image.Image, image_key: str | None = None, target=None, strategy: str | None = None) -> tuple[Image.Image, dict]:
    """Crop for the embedding model: target garment on white, with a padding margin.
    Falls back to the full image (with the reason) when segmentation is not reliable."""
    s = get_settings()
    t = time.perf_counter()
    res = segment(img, image_key, target, strategy)
    info = res.to_dict()
    tg = res.target
    if tg is None or tg.mask is None:
        if tg is not None and tg.mask is None and valid_bbox(tg.bbox, *img.size):  # box-only detection
            box = _pad(tg.bbox, img.size, s.isolation_padding)
            info.update(method_detail="box crop (no mask)")
            return img.crop(box), info | {"isolate_ms": round((time.perf_counter() - t) * 1000, 1)}
        return img, info | {"isolate_ms": round((time.perf_counter() - t) * 1000, 1)}
    w, h = img.size
    alpha = tg.alpha if tg.alpha is not None else tg.mask.astype(np.float32)
    bbox = tg.bbox
    if target is None and s.isolation_default == "all_garments" and len(res.instances) > 1:
        # no garment named: keep every garment region (a colour split of ONE multi-colour garment, e.g. a
        # lehenga with a contrast border, must not be cut apart). A named target crops just that piece.
        for g in res.instances:
            if g is tg or g.mask is None:
                continue
            alpha = np.maximum(alpha, g.alpha if g.alpha is not None else g.mask.astype(np.float32))
        bbox = bbox_of(alpha > 0.5) or bbox
        info["crop_scope"] = "all_garments"
    else:
        info["crop_scope"] = "target"
    if float((alpha > 0.5).mean()) > 0.92:
        info.update(isolate_ms=round((time.perf_counter() - t) * 1000, 1), crop_scope="none",
                    crop_note="garment already fills the photo; full image used")
        return img, info
    # keep fringes / tassels: grow the alpha by a few pixels, never shrink it
    k = max(3, int(0.008 * max(w, h)) | 1)
    alpha = np.maximum(alpha, cv2.dilate((alpha > 0.5).astype(np.float32), np.ones((k, k), np.uint8)) * 0.999)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    out = arr * alpha[..., None] + 255.0 * (1 - alpha[..., None])
    box = _pad(bbox, (w, h), s.isolation_padding)
    crop = Image.fromarray(out.clip(0, 255).astype(np.uint8)).crop(box)
    info.update(isolate_ms=round((time.perf_counter() - t) * 1000, 1), crop_box=list(box))
    return crop, info


def _pad(b, size, pad: float):
    w, h = size
    x0, y0, x1, y1 = b
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)
