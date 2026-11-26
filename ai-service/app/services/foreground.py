"""Weight-free foreground (garment) estimation.

Used for colour/pattern descriptors and for the default `foreground` garment-isolation mode.
No neural network and no download: it must work on any laptop.

Cues, combined in this order:
  1. background model  - colours seen on the image border (two clusters, so gradient / two-tone
                         backdrops work); pixels far from every border colour are garment
  2. shadow removal    - pixels that are only DARKER than the background with the same hue
                         (cast shadows) are treated as background
  3. edge evidence     - for white-on-white or low-contrast photos the colour cue finds almost
                         nothing; closed edge contours are filled to recover the garment outline
  4. main region       - small disconnected blobs away from the centre (bags, jewellery, props)
                         are removed
  5. GrabCut (optional, refine=True) - OpenCV's graph-cut segmentation initialised from the
                         estimate above; sharper boundaries, ~20-60 ms at 256 px
  6. fallback          - the original single-colour method, then a centred ellipse
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

WORK_SIZE = 256
# Catalogue features and the cv-baseline encoder stay on fg-v2 so existing stored features/embeddings remain
# valid (no forced recompute). Query-time segmentation uses fg-v3 (low-contrast retry, noise-scaled
# backdrop test). Switch catalogue features later with CATALOGUE_FOREGROUND=v3 + recompute_features.py.
FOREGROUND_VERSION = "fg-v2"
FOREGROUND_VERSION_LATEST = "fg-v3"


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (..., 3) -> float LAB in the usual ranges (L 0-100, a/b ~ -128..127)."""
    flat = rgb.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(flat, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(rgb.shape)
    lab[..., 0] = lab[..., 0] * (100.0 / 255.0)
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


@dataclass
class Foreground:
    rgb: np.ndarray        # working-size RGB uint8
    mask: np.ndarray       # bool garment mask
    method: str            # which cue produced the mask
    confidence: float      # 0..1 heuristic quality of the mask


def _small(img: Image.Image, size: int) -> np.ndarray:
    small = img.copy()
    small.thumbnail((size, size), Image.Resampling.BILINEAR)
    return np.asarray(small.convert("RGB"), dtype=np.uint8)


def _border(lab: np.ndarray) -> np.ndarray:
    h, w = lab.shape[:2]
    b = max(2, min(h, w) // 25)
    return np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                           lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])


def _clean(mask: np.ndarray, k: int = 5) -> np.ndarray:
    m = mask.astype(np.uint8) * 255
    kernel = np.ones((k, k), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
    # fill interior holes (e.g. a white motif on the garment that matches the backdrop)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    return filled > 0


def _main_region(mask: np.ndarray) -> np.ndarray:
    """Keep the largest component plus components that are big or near the centre."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 2:
        return mask
    h, w = mask.shape
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    keep = np.zeros(n, bool)
    keep[largest] = True
    for i in range(1, n):
        cx, cy = cents[i]
        central = abs(cx - w / 2) < w * 0.25 and abs(cy - h / 2) < h * 0.3
        if areas[i - 1] >= 0.25 * areas[largest - 1] or (central and areas[i - 1] >= 0.05 * areas[largest - 1]):
            keep[i] = True
    return keep[labels]


def _colour_mask(lab: np.ndarray, thresh_override: float | None = None, v2: bool = False) -> tuple[np.ndarray, float, bool]:
    """Returns (mask, border spread, fills_frame). fills_frame = the dominant border colour also
    dominates the image centre, i.e. the garment itself touches the edges (close-up / partial photo)."""
    border = _border(lab)
    h, w = lab.shape[:2]
    centre = lab[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)].reshape(-1, 3)
    # two background clusters (gradient / two-tone backdrops, floor + wall)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    k = 2 if len(border) >= 20 else 1
    cv2.setRNGSeed(0)  # OpenCV k-means uses a global RNG: seed it so the same photo always gives the same mask
    _, labels, centers = cv2.kmeans(border.astype(np.float32), k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    spreads = []
    dist = np.full(lab.shape[:2], np.inf, np.float32)
    fills_frame = False
    for c in range(k):
        members = border[labels.ravel() == c]
        share = len(members) / len(border)
        if share < 0.1:
            continue
        sp = float(np.median(np.linalg.norm(members - centers[c], axis=1)))
        spreads.append(sp)
        # "same colour as the backdrop" must respect the backdrop's own noise: on a perfectly uniform
        # black backdrop a near-black garment is still different (scale with spread, floor 4)
        near_centre = float((np.linalg.norm(centre - centers[c], axis=1) < (max(10.0, 2.5 * sp) if v2 else max(4.0, 3.0 * sp))).mean())
        if share >= 0.3 and near_centre > 0.5 and float(np.linalg.norm(centers[c] - [100, 0, 0])) > 8:
            # this "background" colour fills the centre too -> it is the garment, not a backdrop
            # (a near-white backdrop is excluded: that is the white-on-white case handled by edges)
            fills_frame = True
            continue
        d = np.linalg.norm(lab - centers[c], axis=2)
        # shadows: same hue as the backdrop, just darker -> shrink their distance
        dl = lab[..., 0] - centers[c][0]
        dab = np.linalg.norm(lab[..., 1:] - centers[c][1:], axis=2)
        shadow = (dl < 0) & (dl > -35) & (dab < 6)
        d = np.where(shadow, dab, d)
        dist = np.minimum(dist, d)
    spread = float(np.median(spreads)) if spreads else 5.0
    thresh = thresh_override if thresh_override is not None else max(10.0, 2.5 * spread)
    return dist > thresh, spread, fills_frame


def _border_touch(mask: np.ndarray) -> float:
    return float(np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).mean())


def _edge_mask(lab: np.ndarray) -> np.ndarray:
    L = np.clip(lab[..., 0] * 2.55, 0, 255).astype(np.uint8)
    L = cv2.GaussianBlur(L, (3, 3), 0)
    med = float(np.median(L))
    edges = cv2.Canny(L, max(5, int(0.33 * med * 0.2)), max(15, int(0.66 * med * 0.35)))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    return _clean(edges > 0, k=3)


def _nested(lab: np.ndarray, mask: np.ndarray, method: str) -> tuple[np.ndarray, str]:
    """Product photo inside a larger scene (e.g. a catalogue shot on a table, a screenshot on a
    page): the first mask is the whole inner photo. If that region has its own uniform backdrop,
    segment again inside it."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if (y1 - y0) < 32 or (x1 - x0) < 32:
        return mask, method
    sub = lab[y0:y1, x0:x1]
    border = _border(sub)
    bg = np.median(border, axis=0)
    spread = float(np.median(np.linalg.norm(border - bg, axis=1)))
    if spread > 4:  # inner region has no uniform backdrop -> it is the garment itself
        return mask, method
    inner = _clean(np.linalg.norm(sub - bg, axis=2) > max(10.0, 2.5 * spread)) & mask[y0:y1, x0:x1]
    if not 0.1 <= inner.mean() <= 0.9:
        return mask, method
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = _main_region(inner)
    return out, method + "+nested"


def _fill_outline(mask: np.ndarray) -> np.ndarray:
    """Fill holes enclosed by a garment outline (closed contours only; an outline that runs off the
    frame is left as it is - the border check below then falls back instead of guessing)."""
    import cv2

    h, w = mask.shape
    pad = np.ones((h + 2, w + 2), np.uint8)  # 1 = background (frame included), 0 = outline
    pad[1:-1, 1:-1] = (~mask).astype(np.uint8)
    cv2.floodFill(pad, None, (0, 0), 2)      # background reachable from outside -> 2; enclosed holes stay 1
    return (pad[1:-1, 1:-1] != 2) | mask


def _ellipse(h: int, w: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((yy - h / 2) / (h * 0.42)) ** 2 + ((xx - w / 2) / (w * 0.38)) ** 2 <= 1.0


def _grabcut(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = mask.shape
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    gc[mask] = cv2.GC_PR_FGD
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    gc[inner] = cv2.GC_FGD
    b = max(2, min(h, w) // 40)
    gc[:b, :] = gc[-b:, :] = cv2.GC_BGD
    gc[:, :b] = gc[:, -b:] = cv2.GC_BGD
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.setRNGSeed(0)  # GrabCut initialises its colour models with k-means -> deterministic
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    out = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    return _clean(out)


def estimate_foreground(img: Image.Image, size: int = WORK_SIZE, refine: bool = False, version: str = "v3") -> Foreground:
    """version="v2" reproduces the v2 behaviour exactly (catalogue features, cv-baseline vectors)."""
    v2 = version == "v2"
    rgb = _small(img, size)
    lab = rgb_to_lab(rgb)
    h, w = lab.shape[:2]
    cmask, spread, fills_frame = _colour_mask(lab, v2=v2)
    if fills_frame and cmask.mean() > 0.95:
        # close-up / partial photo: the garment colour fills the border AND the centre and no other
        # backdrop colour exists, so the whole frame is garment. (If a second backdrop colour exists,
        # e.g. white at the sides, the normal path below already separates garment from it.)
        return Foreground(rgb, np.ones_like(cmask), "full_frame", 0.6)
    mask = _main_region(_clean(cmask))
    frac = float(mask.mean())
    method, conf = "colour", 0.85 if spread < 6 else 0.65   # busy backdrops lower confidence
    if 0.05 <= frac <= 0.9:
        mask, method = _nested(lab, mask, method)

    if not v2 and frac < 0.05 and spread < 3.0:
        # very uniform backdrop but almost nothing found: low-contrast garment (black on near-black,
        # navy on black). Retry with a threshold scaled to the backdrop's own noise level.
        cm2, _, _ = _colour_mask(lab, thresh_override=max(4.0, 3.0 * spread))
        lmask = _main_region(_clean(cm2))
        lfrac = float(lmask.mean())
        if 0.05 <= lfrac <= 0.95 and _border_touch(lmask) < 0.15:
            mask, frac, method, conf = lmask, lfrac, "colour_lowcontrast", 0.6
    if frac < 0.05:
        # very low contrast (e.g. white dress on white): try the outline instead
        emask = _main_region(_edge_mask(lab))
        if not v2:
            emask = _fill_outline(emask)  # an outline is not a garment: fill the closed contour
        efrac = float(emask.mean())
        if 0.05 <= efrac <= 0.95 and _border_touch(emask) < 0.15:
            mask, frac, method, conf = emask, efrac, "edges", 0.55
    if frac < 0.05 or frac > 0.97:
        mask, method, conf = _ellipse(h, w), "centre_ellipse", 0.2
    elif refine:
        gmask = _grabcut(rgb, mask)
        if gmask is not None and 0.05 <= gmask.mean() <= 0.97:
            # accept GrabCut only if it broadly agrees with the estimate (IoU), else keep the estimate
            inter = float((gmask & mask).sum()); union = float((gmask | mask).sum()) or 1.0
            if inter / union > 0.5:
                if not v2:
                    # GrabCut cannot keep pixels on the image border: restore the estimate there so a
                    # garment that runs off the frame (close-up / partial photo) keeps its edge rows
                    band = np.zeros_like(mask)
                    b = max(2, int(0.03 * max(h, w)))
                    band[:b, :] = band[-b:, :] = band[:, :b] = band[:, -b:] = True
                    gmask = gmask | (mask & band)
                mask, method, conf = gmask, method + "+grabcut", min(0.95, conf + 0.1)
    return Foreground(rgb, mask, method, conf)


def foreground_mask(img: Image.Image, size: int = WORK_SIZE, version: str = "v2") -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible API: (rgb_small uint8 HxWx3, mask bool HxW). Defaults to the v2 behaviour
    because cv-baseline embeddings and catalogue features stored by v2 depend on it."""
    fg = estimate_foreground(img, size, version=version)
    return fg.rgb, fg.mask


def legacy_foreground_mask(img: Image.Image, size: int = WORK_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """The original v1 single-colour method, kept for comparison and as a reference."""
    rgb = _small(img, size)
    lab = rgb_to_lab(rgb)
    border = _border(lab)
    bg = np.median(border, axis=0)
    border_spread = float(np.median(np.linalg.norm(border - bg, axis=1)))
    dist = np.linalg.norm(lab - bg, axis=2)
    mask = _clean(dist > max(12.0, 2.5 * border_spread))
    if mask.mean() < 0.05 or mask.mean() > 0.97:
        mask = _ellipse(*mask.shape)
    return rgb, mask


def crop_to_mask(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return rgb, mask
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum()); union = float((a | b).sum())
    return inter / union if union else 0.0
