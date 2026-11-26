"""Weight-free garment isolation (always available, CPU, a few ms).

* scene analysis - is this a product-only photo, a person wearing clothes, several garments, or hard?
* person path    - foreground minus skin, face and hair -> clothing
* instance split - separate garments by colour + connectivity (kurti / trousers / dupatta)

Limits (honest): skin is found by colour, so a garment in the wearer's skin tone is ambiguous - the
result then gets a LOW confidence instead of a confident wrong mask. Learned parsers (SegFormer /
SCHP, see parsing.py) are far more reliable on real photos when installed.
"""
from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from app.config import get_settings
from app.services.colour import ciede2000
from app.services.foreground import Foreground, rgb_to_lab
from app.services.segmentation.refine import fill_holes, occlusion_ratio, remove_specks
from app.services.segmentation.types import GarmentInstance, bbox_of


# ------------------------------------------------------------------ faces (OpenCV Haar cascade, BSD licence)
@lru_cache(maxsize=1)
def _face_cascade():
    if not hasattr(cv2, "CascadeClassifier"):
        return None  # OpenCV 5 moved Haar cascades to opencv-contrib; the head-blob heuristic is used instead
    s = get_settings()
    candidates = [s.face_cascade_path] if s.face_cascade_path else []
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", "") or ""
    candidates += [os.path.join(data_dir, "haarcascade_frontalface_default.xml"),
                   str(s.model_cache_path / "opencv" / "haarcascade_frontalface_default.xml")]
    for p in candidates:
        if p and os.path.exists(p):
            c = cv2.CascadeClassifier(p)
            if not c.empty():
                return c
    return None


def face_detector_available() -> bool:
    return _face_cascade() is not None


def detect_faces(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    c = _face_cascade()
    if c is None:
        return []
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    m = max(20, min(rgb.shape[:2]) // 14)
    faces = c.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(m, m))
    return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in faces]


# ------------------------------------------------------------------ skin
def skin_mask(rgb: np.ndarray, faces: list | None = None) -> np.ndarray:
    """Skin pixels. With a detected face the model adapts to THAT person's skin tone (LAB distance to
    the face centre); otherwise classic YCrCb + HSV rules covering light to dark skin."""
    if faces:
        x0, y0, x1, y1 = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
        cx0, cx1 = x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4
        cy0, cy1 = y0 + (y1 - y0) // 3, y1 - (y1 - y0) // 6
        sample = rgb_to_lab(rgb[cy0:cy1, cx0:cx1].reshape(1, -1, 3))[0]
        if len(sample) > 20:
            ref = np.median(sample, axis=0)
            lab = rgb_to_lab(rgb)
            d = ciede2000(lab.reshape(-1, 3), ref).reshape(rgb.shape[:2])
            return d < 13
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    cr, cb = ycc[..., 1].astype(int), ycc[..., 2].astype(int)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    rule_ycc = (cr >= 135) & (cr <= 180) & (cb >= 80) & (cb <= 130)
    rule_hsv = ((h <= 25) | (h >= 165)) & (s >= 40) & (s <= 190) & (v >= 50)
    return rule_ycc & rule_hsv


def head_adapted_skin(rgb: np.ndarray, person_mask: np.ndarray, generic: np.ndarray) -> np.ndarray | None:
    """No face detector: the generic skin rule also fires on beige / pink fabric and walls. Take the
    wearer's own skin tone from the head zone of the person mask (top 13 %, central half) and keep
    only pixels close to it (CIEDE2000 < 13). None when the head zone shows no skin (e.g. back view)."""
    pb = bbox_of(person_mask)
    if pb is None:
        return None
    x0, y0, x1, y1 = pb
    pw, ph = x1 - x0, y1 - y0
    zone = np.zeros_like(person_mask)
    zone[y0:y0 + max(2, int(0.13 * ph)), x0 + pw // 4:x1 - pw // 4] = True
    sel = zone & person_mask & generic
    if sel.sum() < 12:
        return None
    ref = np.median(rgb_to_lab(rgb[sel].reshape(1, -1, 3))[0], axis=0)
    d = ciede2000(rgb_to_lab(rgb).reshape(-1, 3), ref).reshape(rgb.shape[:2])
    return d < 13


def _head_and_hair(rgb: np.ndarray, skin: np.ndarray, person: np.ndarray, faces: list) -> np.ndarray:
    """Face + hair region to exclude. From the face box if detected, else from the top skin blob."""
    h, w = skin.shape
    out = np.zeros_like(skin)
    boxes = list(faces)
    if not boxes:
        pb = bbox_of(person)
        if pb is None:
            return out
        top_limit = pb[1] + 0.22 * (pb[3] - pb[1])
        n, lab, stats, cents = cv2.connectedComponentsWithStats(skin.astype(np.uint8), 8)
        for i in range(1, n):
            if cents[i][1] < top_limit and stats[i, cv2.CC_STAT_AREA] > 0.002 * h * w:
                x, y, bw, bh = stats[i, :4]
                boxes.append((x, y, x + bw, y + bh))
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    for x0, y0, x1, y1 in boxes:
        fw, fh = x1 - x0, y1 - y0
        # hair: dark pixels in a region around and above the face
        hx0, hx1 = max(0, int(x0 - 0.35 * fw)), min(w, int(x1 + 0.35 * fw))
        hy0, hy1 = max(0, int(y0 - 0.6 * fh)), min(h, int(y1 + 0.1 * fh))
        region = np.zeros_like(skin)
        region[hy0:hy1, hx0:hx1] = True
        out |= region & ((lum < 70) | skin)
        out[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return out


# ------------------------------------------------------------------ scene analysis
def analyze_scene(rgb: np.ndarray, fg: Foreground) -> dict:
    """Cheap signals that decide which processing path is worth running."""
    faces = detect_faces(rgb)
    mask = fg.mask
    skin = skin_mask(rgb, faces)
    fg_px = max(1, int(mask.sum()))
    skin_in_fg = float((skin & mask).sum() / fg_px)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True) if n > 1 else []
    big = [a for a in areas if areas and a >= 0.15 * areas[0]]
    head_blob = False
    pb = bbox_of(mask)
    if pb is not None and skin.any():
        # a head = compact, roughly round, uniform skin blob at the TOP of the figure, horizontally central
        # (skin-coloured print on a garment is textured, scattered or not at the top)
        pw, ph = pb[2] - pb[0], pb[3] - pb[1]
        # look only in the top band of the figure, so a head is not merged with arms/legs into one blob
        band = np.zeros_like(mask)
        band[pb[1]:pb[1] + int(0.28 * ph), :] = True
        n2, lab2, st2, c2 = cv2.connectedComponentsWithStats((skin & mask & band).astype(np.uint8), 8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        for i in range(1, n2):
            x, y, bw, bh, area = st2[i]
            if area < 0.003 * mask.size or y > pb[1] + 0.12 * ph or abs(c2[i][0] - (pb[0] + pw / 2)) > 0.2 * pw:
                continue
            # a head is SMALL relative to the figure: a skin-toned garment (beige/peach dress) is not a head
            if bh > 0.25 * ph or area > 0.12 * fg_px:
                continue
            comp = lab2 == i
            fill = area / float(bw * bh)
            core = cv2.erode(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # ignore blurred edge pixels
            tex = float(gray[core].std()) if core.sum() > 20 else float(gray[comp].std())
            if 0.55 <= fill and 0.6 <= bw / max(1, bh) <= 1.6 and tex < 14:
                head_blob = True
                break
    person = bool(faces) or (skin_in_fg >= 0.05 and head_blob)
    # most of the "garment" is skin-coloured: either a skin-toned garment or bare skin - colour alone
    # cannot tell, so the photo is marked ambiguous (low confidence, no confident mask)
    ambiguous_skin = skin_in_fg > 0.5
    return {"ambiguous_skin": ambiguous_skin,"faces": len(faces), "face_boxes": faces, "skin_share": round(skin_in_fg, 3), "head_blob": head_blob,
            "components": len(big), "fg_method": fg.method, "fg_confidence": round(fg.confidence, 2),
            "garment_share": round(float(mask.mean()), 3), "person": person, "_skin": skin}


# ------------------------------------------------------------------ instance split (colour + connectivity)
def split_instances(rgb: np.ndarray, clothing: np.ndarray, source: str, ref_box=None,
                    min_share: float = 0.06, max_k: int = 3) -> list[GarmentInstance]:
    """Separate garments of different colours (kurti / trousers / dupatta) and disconnected pieces."""
    if not clothing.any():
        return []
    lab = rgb_to_lab(rgb)
    px = lab[clothing]
    if len(px) > 6000:
        px = px[np.random.default_rng(0).choice(len(px), 6000, replace=False)]
    k = min(max_k, max(1, len(np.unique(px.round(), axis=0))))
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(px.astype(np.float32), k, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 2, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(float)
    # merge colour clusters that are perceptually the same (shading of one fabric)
    groups = list(range(k))
    for i in range(k):
        for j in range(i + 1, k):
            if float(ciede2000(centers[i], centers[j])) < 12:
                groups = [groups[i] if g == groups[j] else g for g in groups]
    full = lab[clothing]
    d = np.stack([np.linalg.norm(full - c, axis=1) for c in centers], 1)
    assign = np.array(groups)[d.argmin(1)]
    cmap = np.full(clothing.shape, -1, int)
    cmap[clothing] = assign
    total = float(clothing.sum())
    pb = ref_box or bbox_of(clothing)
    inst = []
    for g in sorted(set(groups)):
        gm = cmap == g
        n, labm, stats, _ = cv2.connectedComponentsWithStats(gm.astype(np.uint8), 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_share * total:
                m = labm == i
                inst.append(GarmentInstance("garment region", _group_from_position(m, pb), 0.5, bbox_of(m), m, source))
    if len(inst) <= 1:
        m = remove_specks(clothing)
        return [GarmentInstance("garment region", "garment", 0.5, bbox_of(m), m, source)] if m.any() else []
    inst.sort(key=lambda x: -x.area)
    return inst


def _group_from_position(m: np.ndarray, ref_box) -> str:
    b = bbox_of(m)
    if b is None or ref_box is None:
        return "garment"
    top, bottom = ref_box[1], ref_box[3]
    hgt = max(1, bottom - top)
    t, bt = (b[1] - top) / hgt, (b[3] - top) / hgt
    if bt - t > 0.7:
        return "full_body"
    if bt <= 0.7:
        return "upper"
    if t >= 0.4:
        return "lower"
    return "garment"


# ------------------------------------------------------------------ person path
def person_clothing(rgb: np.ndarray, person_mask: np.ndarray, scene: dict) -> tuple[list[GarmentInstance], float, dict]:
    """Clothing = person region minus skin, face and hair. Returns (instances, confidence, details)."""
    faces = scene.get("face_boxes") or []
    skin = scene.get("_skin")
    if skin is None:
        skin = skin_mask(rgb, faces)
    adapted = None if faces else head_adapted_skin(rgb, person_mask, skin)
    if adapted is not None:
        skin = adapted
    skin_d = cv2.dilate(skin.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    head = _head_and_hair(rgb, skin, person_mask, faces)
    clothing = person_mask & ~skin_d & ~head
    clothing = remove_specks(clothing, min_area_frac=0.004)
    removed = float((person_mask & ~clothing).sum() / max(1, person_mask.sum()))
    share = float(clothing.sum() / max(1, person_mask.sum()))
    conf = 0.62
    details = {"removed_skin_face_hair_share": round(removed, 3), "clothing_share_of_person": round(share, 3)}
    if not 0.2 <= share <= 0.95:
        conf -= 0.2
        details["note"] = "unusual clothing share - result may be unreliable"
    # garment in the wearer's skin tone? -> ambiguous, say so with a low confidence
    if clothing.any() and skin.any():
        g = np.median(rgb_to_lab(rgb[clothing].reshape(1, -1, 3))[0], axis=0)
        s = np.median(rgb_to_lab(rgb[skin].reshape(1, -1, 3))[0], axis=0)
        if float(ciede2000(g, s)) < 14:
            conf = min(conf, 0.35)
            details["note"] = "garment colour is close to skin colour; skin removal is ambiguous"
    if not faces:
        conf -= 0.05  # no face confirmation: skin tone from generic rules
    inst = split_instances(rgb, clothing, "classical_person", ref_box=bbox_of(person_mask))
    rescued = _skin_toned_lower(skin & person_mask & ~head, person_mask)
    occl = occlusion_ratio(fill_holes(clothing) & person_mask, skin_d) if clothing.any() else 0.0
    for i in inst:
        i.confidence = round(max(0.05, conf - min(0.3, occl)), 3)
        i.occlusion = round(occl, 3)
    if rescued is not None:
        # legs are thin bars; a single wide skin-toned block below the waist is far more likely trousers or a
        # skirt in the wearer's skin tone. Reported separately with a low confidence - never merged silently.
        inst.append(GarmentInstance("garment region (skin-toned)", "lower", 0.35, bbox_of(rescued), rescued,
                                    "classical_person"))
        details["note"] = "a skin-toned lower garment was kept with low confidence; a parsing model resolves this reliably"
    return inst, max(0.05, conf - min(0.3, occl)), details


def _skin_toned_lower(cand: np.ndarray, person_mask: np.ndarray) -> np.ndarray | None:
    pb = bbox_of(person_mask)
    if pb is None or not cand.any():
        return None
    px0, py0, px1, py1 = pb
    pw, ph = px1 - px0, py1 - py0
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if y < py0 + 0.45 * ph or bw < 0.22 * pw or bh < 0.15 * ph:
            continue
        if a / float(bw * bh) < 0.6 or a < 0.06 * person_mask.sum():
            continue
        return remove_specks(lab == i, min_area_frac=0.004)
    return None


def product_instances(rgb: np.ndarray, fg: Foreground) -> list[GarmentInstance]:
    """Garments without a person: one instance per disconnected or differently coloured piece."""
    return split_instances(rgb, fg.mask, "foreground", min_share=0.08)


def remove_occluders(rgb: np.ndarray, mask: np.ndarray, skin: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Uniform skin-coloured blobs INSIDE a garment (a hand in front of it) are not garment pixels.
    They are removed - never painted over - and the occluded share is returned so confidence can drop.
    Textured skin-tone prints are kept (the blob must be smooth and compact)."""
    skin = skin_mask(rgb) if skin is None else skin
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    cand = skin & mask
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out, removed = mask.copy(), 0
    area = max(1, int(mask.sum()))
    hits = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        comp = lab == i
        if a < 0.03 * area or a > 0.5 * area:
            continue
        if max(bw, bh) > 4 * min(bw, bh):       # stripes / bands are print, not a hand
            continue
        if (comp & inner).sum() < 0.6 * a:     # must sit inside the garment, not on its edge
            continue
        if a / float(bw * bh) < 0.5 or float(gray[comp].std()) > 14:
            continue
        hits.append((comp, a))
    if len(hits) >= 3:                          # a repeated skin-tone motif is a print
        return out, 0.0
    for comp, a in hits:
        out &= ~comp
        removed += a
    return out, removed / float(area)


def to_image(fg_mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.asarray(Image.fromarray(fg_mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127
