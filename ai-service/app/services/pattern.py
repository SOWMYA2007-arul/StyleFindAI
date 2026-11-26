"""Texture / print descriptor used for "similar pattern" ranking.

We describe the garment surface with Local Binary Patterns (LBP: how each pixel compares to
its neighbours - captures stripes, checks, fine prints), gradient-orientation histograms
(stripes have one dominant direction, checks have two) and edge density (solid fabrics have
few edges inside the garment). This is a classical, explainable descriptor, not a learned one.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

from app.services.foreground import crop_to_mask, foreground_mask

FEATURE_VERSION = "pattern-v1"


def _inner_region(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Erode the mask so garment outlines don't count as 'pattern' edges."""
    m = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1) > 0
    return m if m.sum() > 200 else mask


def pattern_descriptor(img: Image.Image, fg=None) -> tuple[np.ndarray, str]:
    rgb, mask = (fg.rgb, fg.mask) if fg is not None else foreground_mask(img)
    rgb, mask = crop_to_mask(rgb, mask)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if min(gray.shape) < 16:
        gray = cv2.resize(gray, (64, 64)); mask = np.ones_like(gray, bool)
    inner = _inner_region(gray, mask)

    feats = []
    for p, r in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(gray, p, r, method="uniform")
        hist, _ = np.histogram(lbp[inner], bins=p + 2, range=(0, p + 2))
        feats.append(hist / max(1, hist.sum()))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = np.mod(ang, 180.0)
    strong = inner & (mag > 30)
    ohist, _ = np.histogram(ang[strong], bins=8, range=(0, 180), weights=mag[strong])
    ohist = ohist / max(1e-6, ohist.sum())
    feats.append(ohist)

    edges = cv2.Canny(gray, 60, 140) > 0
    edge_density = float(edges[inner].mean()) if inner.any() else 0.0
    feats.append(np.array([min(1.0, edge_density * 4)]))

    vec = np.concatenate(feats).astype(np.float32)
    label = "solid" if edge_density < 0.04 else "patterned"
    return vec, label


def pattern_similarity(a: np.ndarray | list | None, b: np.ndarray | list | None) -> float:
    """Histogram intersection-style similarity in [0,1]. NaN if either side is missing."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return float("nan")
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return float("nan")
    return _block_affinity(a, b)


def _block_affinity(a: np.ndarray, b: np.ndarray) -> float:
    # Bhattacharyya (Hellinger) affinity per histogram block - robust for normalised histograms.
    # Blocks: LBP8 (10), LBP16 (18), orientation (8), edge density (1)
    idx = [(0, 10), (10, 28), (28, 36)]
    sims = [float(np.sum(np.sqrt(np.clip(a[s:e], 0, None) * np.clip(b[s:e], 0, None)))) for s, e in idx]
    sims.append(1.0 - abs(float(a[-1]) - float(b[-1])))
    w = [0.3, 0.3, 0.2, 0.2]
    return float(np.clip(sum(wi * si for wi, si in zip(w, sims)), 0.0, 1.0))


# --------------------------------------------------------------------------- pattern classification
PATTERN_CLASSIFIER_VERSION = "pattern-cls-v1"


def _dominant_orientations(gray: np.ndarray, inner: np.ndarray) -> tuple[np.ndarray, float]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = np.mod(ang, 180.0)
    strong = inner & (mag > max(20.0, float(np.percentile(mag[inner], 75)) if inner.any() else 20.0))
    hist, _ = np.histogram(ang[strong], bins=18, range=(0, 180), weights=mag[strong])
    total = float(hist.sum())
    return (hist / total if total > 0 else hist), float(strong.mean() if strong.size else 0.0)


def _periodicity(gray: np.ndarray, inner: np.ndarray) -> float:
    """How periodic the texture is (0..1): strength of the largest non-DC peak of the 2-D spectrum.
    Stripes, checks and dots are periodic; florals and abstract prints much less so."""
    g = gray.astype(np.float32)
    g = np.where(inner, g - g[inner].mean() if inner.any() else 0.0, 0.0)
    f = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    h, w = f.shape
    f[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3] = 0  # remove DC / very low frequencies
    total = float(f.sum()) or 1.0
    top = np.sort(f.ravel())[-8:].sum()
    return float(min(1.0, (top / total) * 25))


def _blob_stats(gray: np.ndarray, inner: np.ndarray, edges: np.ndarray | None = None) -> tuple[int, float, float, float]:
    """Round, similar-sized, evenly spaced blobs = polka dots.
    Returns (count, size coefficient of variation, median neighbour-distance / blob-size ratio,
    share of the garment's edge pixels that lie on the dots - near 1 for true polka dots, low when
    dots are only one element of a busier print such as flower centres).
    Flower petals and check gaps also form blobs, but petals touch each other (ratio ~1) and
    squares/diamonds fail the circularity test."""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea, params.minArea, params.maxArea = True, 6, 900
    params.filterByCircularity, params.minCircularity = True, 0.83
    params.filterByConvexity, params.minConvexity = True, 0.9
    params.filterByInertia, params.minInertiaRatio = True, 0.6
    det = cv2.SimpleBlobDetector_create(params)
    pts, sizes = [], []
    for img in (gray, 255 - gray):  # dark dots on light fabric and light dots on dark fabric
        for kp in det.detect(img):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= y < inner.shape[0] and 0 <= x < inner.shape[1] and inner[y, x]:
                pts.append(kp.pt); sizes.append(kp.size)
    if len(sizes) < 3:
        return len(sizes), 1.0, 0.0, 0.0
    sizes, pts = np.array(sizes), np.array(pts)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    coverage = 0.0
    if edges is not None:
        near = np.zeros_like(inner, dtype=np.uint8)
        for (x, y), sz in zip(pts, sizes):
            cv2.circle(near, (int(x), int(y)), int(sz / 2 + 3), 1, -1)
        e = edges & inner
        coverage = float((e & (near > 0)).sum() / max(1, e.sum()))
    return len(sizes), float(sizes.std() / max(1e-6, sizes.mean())), float(np.median(nn) / max(1e-6, np.median(sizes))), coverage


def classify_pattern(img: Image.Image, fg=None) -> dict:
    """Weight-free pattern classifier -> {"value", "confidence", "source", "evidence"}.

    Recognises solid / striped / checked / polka dot from measurable texture evidence. Anything
    else (floral, paisley, abstract, embroidery ...) is left to the zero-shot model; here it is
    'unknown' rather than guessed, as required.
    """
    rgb, mask = (fg.rgb, fg.mask) if fg is not None else foreground_mask(img)
    rgb, mask = crop_to_mask(rgb, mask)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if min(gray.shape) < 24:
        return {"value": "unknown", "confidence": 0.0, "source": "texture", "evidence": "garment region too small"}
    inner = _inner_region(gray, mask)
    lab_l = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    l_std = float(lab_l[inner].std()) if inner.any() else 0.0
    edges = cv2.Canny(gray, 50, 130) > 0
    edge_density = float(edges[inner].mean()) if inner.any() else 0.0
    ohist, _ = _dominant_orientations(gray, inner)
    period = _periodicity(gray, inner)
    n_blobs, blob_cv, spacing, coverage = _blob_stats(gray, inner, edges)
    area = max(1, int(inner.sum()))
    blob_density = n_blobs / (area / 1000.0)
    # Low-resolution images (e.g. 60x80 catalogue thumbnails) carry little texture information.
    res_factor = float(min(1.0, min(gray.shape) / 120.0))
    ev = {"edge_density": round(edge_density, 4), "luma_std": round(l_std, 2), "periodicity": round(period, 3),
          "blobs": n_blobs, "blob_size_cv": round(blob_cv, 2), "blob_spacing": round(spacing, 2),
          "blob_edge_coverage": round(coverage, 2),
          "resolution_factor": round(res_factor, 2)}

    def out(value, conf):
        conf = float(np.clip(conf * (0.5 + 0.5 * res_factor), 0.0, 1.0))
        return {"value": value if conf >= 0.5 else "unknown", "confidence": round(conf, 3), "source": "texture",
                "evidence": ev, "candidate": value}

    if edge_density < 0.025:
        # few edges inside the garment: solid fabric (shading / a border line lowers confidence)
        if l_std < 12:
            return out("solid", 0.9 - edge_density * 8)
        if l_std < 25:
            return out("solid", 0.62)
        return {"value": "unknown", "confidence": 0.0, "source": "texture", "evidence": ev}
    # orientation peaks: merge each 10-degree bin with its neighbours
    smooth = ohist + np.roll(ohist, 1) + np.roll(ohist, -1)
    i1 = int(np.argmax(smooth)); p1 = float(smooth[i1])
    smooth2 = smooth.copy()
    for d in (-2, -1, 0, 1, 2):
        smooth2[(i1 + d) % 18] = 0
    i2 = int(np.argmax(smooth2)); p2 = float(smooth2[i2])
    sep = abs(i1 - i2) * 10
    sep = min(sep, 180 - sep)
    ev.update({"orientation_peak1": round(p1, 3), "orientation_peak2": round(p2, 3), "peak_separation_deg": sep})
    periodic = period > 0.5 and edge_density >= 0.03
    if periodic and p1 > 0.30 and p2 > 0.22 and 70 <= sep <= 110:
        return out("checked", min(0.92, 0.45 + p2 + 0.3 * period))
    if periodic and p1 > 0.55 and p2 < 0.25:
        return out("striped", min(0.92, 0.3 + p1 * 0.6 + 0.2 * period))
    if n_blobs >= 8 and blob_cv < 0.45 and spacing > 1.8 and p1 < 0.45 and coverage > 0.6:
        return out("polka dot", min(0.95, 0.55 + 0.04 * min(n_blobs, 10)))
    if edge_density > 0.06:
        return {"value": "unknown", "confidence": 0.0, "source": "texture", "candidate": "patterned",
                "evidence": ev | {"note": "patterned, but not a recognisable stripe/check/dot structure"}}
    return {"value": "unknown", "confidence": 0.0, "source": "texture", "evidence": ev}
