"""Mask post-processing: detail-preserving clean-up, edge-aware boundary refinement, occlusion
estimate, and the segmentation metrics used by the benchmark.

Why a guided filter? Masks are predicted at low resolution (256-512 px) for speed. Plain resizing
gives blocky edges and deletes 2-3 px details (straps, tassels). The guided filter (He et al. 2010)
upsamples the mask using the full-resolution photo as a guide, so the boundary snaps to real image
edges and thin structures that are visible in the photo survive. It also yields a soft alpha that
represents semi-transparent fabric (dupatta) honestly instead of forcing 0/1.
"""
from __future__ import annotations

import cv2
import numpy as np


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Grey-guide guided filter. guide/src float32 in [0,1], same HxW."""
    mean_i, mean_p = _box(guide, r), _box(src, r)
    corr_ip, corr_ii = _box(guide * src, r), _box(guide * guide, r)
    var_i, cov_ip = corr_ii - mean_i * mean_i, corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box(a, r) * guide + _box(b, r)


def refine_to_image(mask_small: np.ndarray, rgb_full: np.ndarray, radius: int | None = None,
                    eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Upsample a low-res bool mask to the photo's resolution with edge-aware refinement.
    Returns (bool mask, float alpha 0..1)."""
    h, w = rgb_full.shape[:2]
    up = cv2.resize(mask_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    scale = max(h / mask_small.shape[0], w / mask_small.shape[1])
    r = radius or int(np.clip(round(scale * 2), 2, 16))
    guide = cv2.cvtColor(rgb_full, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    alpha = np.clip(guided_filter(guide, up, r, eps), 0.0, 1.0)
    # limit changes to a band around the original boundary: interior/exterior stay as predicted
    band = cv2.dilate((np.abs(up - 0.5) < 0.5).astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    alpha = np.where(band, alpha, up)
    return alpha > 0.5, alpha


def remove_specks(mask: np.ndarray, min_area_frac: float = 0.002, keep_largest_only: bool = False) -> np.ndarray:
    """Delete tiny isolated blobs by AREA (not by morphological opening, which also deletes thin
    straps, fringes and lace edges that are part of the garment)."""
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if keep_largest_only:
        return lab == (int(np.argmax(areas)) + 1)
    keep = np.zeros(n, bool)
    keep[1:] = areas >= max(4, min_area_frac * mask.size)
    return keep[lab]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 255, thickness=cv2.FILLED)
    return out > 0


def occlusion_ratio(mask: np.ndarray, occluder: np.ndarray | None = None) -> float:
    """Share of the garment's filled outline that is NOT garment (holes = something in front of it:
    a hand, a bag). With an explicit occluder mask (e.g. skin) only those pixels count.
    The garment is never 'filled in' - this only lowers confidence."""
    filled = fill_holes(mask)
    area = float(filled.sum())
    if area == 0:
        return 0.0
    holes = filled & ~mask
    if occluder is not None:
        holes &= occluder
    return float(holes.sum() / area)


# ------------------------------------------------------------------ metrics
def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    u = float((pred | gt).sum())
    return float((pred & gt).sum() / u) if u else 1.0


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    s = float(pred.sum() + gt.sum())
    return float(2 * (pred & gt).sum() / s) if s else 1.0


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    tp = float((pred & gt).sum())
    p = tp / float(pred.sum()) if pred.sum() else (1.0 if not gt.any() else 0.0)
    r = tp / float(gt.sum()) if gt.sum() else 1.0
    return p, r


def _boundary(m: np.ndarray) -> np.ndarray:
    m8 = m.astype(np.uint8)
    return (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))) > 0


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: int | None = None) -> float:
    """Boundary F-measure (Perazzi et al. 2016): boundary pixels matched within `tol` pixels.
    Sensitive to thin details and edge accuracy, unlike IoU."""
    tol = tol if tol is not None else max(1, int(round(0.008 * np.hypot(*gt.shape))))
    bp, bg = _boundary(pred), _boundary(gt)
    if not bp.any() and not bg.any():
        return 1.0
    if not bp.any() or not bg.any():
        return 0.0
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    gd = cv2.dilate(bg.astype(np.uint8), k) > 0
    pd = cv2.dilate(bp.astype(np.uint8), k) > 0
    prec = float((bp & gd).sum() / bp.sum())
    rec = float((bg & pd).sum() / bg.sum())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
