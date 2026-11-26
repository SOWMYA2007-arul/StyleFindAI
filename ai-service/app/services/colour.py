"""Colour palette extraction and colour similarity.

Why LAB?  In RGB, the numeric distance between two colours does not match how different
they look to people. CIELAB is designed so that Euclidean distance (Delta-E) roughly
matches perceived difference, which is what a shopper cares about.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from app.services.foreground import foreground_mask, rgb_to_lab

# Reference colours (sRGB) for colour names. Pairs that shoppers distinguish are kept separate:
# maroon/red, wine/burgundy, peach/pink, navy/royal blue, cream/white, mustard/yellow, olive/green.
COLOUR_REFERENCE: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 20), "charcoal": (54, 60, 66), "grey": (128, 128, 128), "silver": (192, 192, 192),
    "white": (250, 250, 250), "off white": (240, 236, 222), "cream": (255, 248, 210), "beige": (225, 205, 170),
    "khaki": (195, 176, 145), "tan": (210, 180, 140), "brown": (120, 72, 40), "coffee brown": (85, 55, 38),
    "rust": (183, 65, 14), "maroon": (110, 15, 30), "wine": (114, 47, 55), "burgundy": (128, 0, 32),
    "red": (205, 30, 40), "coral": (250, 115, 95), "rose": (230, 100, 130), "pink": (245, 160, 190),
    "peach": (255, 200, 160), "orange": (240, 120, 30), "mustard": (210, 165, 30), "yellow": (250, 225, 50),
    "gold": (200, 160, 60), "olive": (110, 115, 40), "lime green": (160, 210, 60), "green": (40, 140, 60),
    "bottle green": (0, 90, 50), "sea green": (46, 139, 87), "mint": (170, 230, 200), "teal": (0, 120, 120),
    "turquoise blue": (64, 200, 210), "sky blue": (135, 200, 235), "blue": (40, 90, 200),
    "royal blue": (40, 70, 190), "navy blue": (25, 30, 80), "lavender": (190, 170, 230), "purple": (110, 50, 140),
    "mauve": (190, 140, 170), "magenta": (200, 30, 140),
}
_NAMES = list(COLOUR_REFERENCE)
_REF_LAB = rgb_to_lab(np.array([[COLOUR_REFERENCE[n] for n in _NAMES]], dtype=np.uint8))[0]

# Common aliases / spellings in dataset metadata and user text.
COLOUR_ALIASES = {
    "navy": "navy blue", "dark blue": "navy blue", "grey melange": "grey", "gray": "grey", "offwhite": "off white",
    "ivory": "cream", "turquoise": "turquoise blue", "aqua": "turquoise blue", "violet": "purple", "skin": "beige",
    "nude": "beige", "taupe": "khaki", "mushroom brown": "brown", "copper": "rust", "bronze": "gold", "steel": "grey",
    "metallic": "silver", "fluorescent green": "lime green", "golden": "gold", "light blue": "sky blue",
    "marron": "maroon", "mehroon": "maroon", "magroon": "maroon", "burgandy": "burgundy", "fuchsia": "magenta",
    "light pink": "pink", "baby pink": "pink", "hot pink": "magenta", "lemon": "yellow", "emerald": "bottle green",
    "dark green": "bottle green", "army green": "olive", "multi": "multicolour", "multicolor": "multicolour",
}


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma et al. 2005). Inputs (..., 3) LAB arrays, broadcastable.
    About 1 = just noticeable, 2-10 = similar shade, > 20 = clearly different colours."""
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, dtype=np.float64), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, dtype=np.float64), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hsum,
                   np.where(np.abs(h1p - h2p) <= 180, hsum / 2, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def normalise_colour_name(name: str | None) -> str | None:
    if not name:
        return None
    n = name.strip().lower()
    n = COLOUR_ALIASES.get(n, n)
    return n if n in COLOUR_REFERENCE else None


def nearest_colour_name(lab: np.ndarray) -> str:
    return _NAMES[int(np.argmin(ciede2000(_REF_LAB, np.asarray(lab)[None, :])))]


def nearest_colour(lab: np.ndarray) -> tuple[str, float]:
    d = ciede2000(_REF_LAB, np.asarray(lab)[None, :])
    i = int(np.argmin(d))
    return _NAMES[i], float(d[i])


def colour_name_lab(name: str) -> np.ndarray | None:
    n = normalise_colour_name(name)
    return None if n is None else _REF_LAB[_NAMES.index(n)]


PALETTE_VERSION = "palette-v2-ciede2000"
MERGE_DELTA_E = 6.0      # clusters closer than this are shades of one colour -> merged
MIN_SHARE = 0.04         # colours covering less than 4% of the garment are ignored (noise, stitching)


def extract_palette(img: Image.Image, k: int = 5, fg=None) -> list[dict]:
    """Dominant colours of the garment region, largest first.

    k-means in CIELAB on garment pixels (background removed by the foreground mask), then
    clusters that are perceptually the same colour (CIEDE2000 < 6) are merged, so one shaded
    red fabric is 'red' and not 'red + dark red'. Each entry: lab, weight (share), name,
    delta_e (distance to the named reference - large values mean the name is approximate).
    """
    rgb, mask = (fg.rgb, fg.mask) if fg is not None else foreground_mask(img)
    pixels = rgb[mask]
    if len(pixels) < 50:
        pixels = rgb.reshape(-1, 3)
    if len(pixels) > 5000:  # subsample for speed; deterministic
        idx = np.random.default_rng(0).choice(len(pixels), 5000, replace=False)
        pixels = pixels[idx]
    lab = rgb_to_lab(pixels.reshape(1, -1, 3))[0]
    k = min(k, max(1, len(np.unique(pixels, axis=0))))
    km = KMeans(n_clusters=k, n_init=2, random_state=0).fit(lab)
    counts = np.bincount(km.labels_, minlength=k).astype(float)
    clusters = [[km.cluster_centers_[i].astype(float), counts[i]] for i in range(k) if counts[i] > 0]
    # greedy merge of perceptually identical clusters (weighted mean in LAB)
    merged = True
    while merged and len(clusters) > 1:
        merged = False
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = float(ciede2000(clusters[i][0], clusters[j][0]))
                if d < MERGE_DELTA_E and (best is None or d < best[0]):
                    best = (d, i, j)
        if best:
            _, i, j = best
            wi, wj = clusters[i][1], clusters[j][1]
            clusters[i] = [(clusters[i][0] * wi + clusters[j][0] * wj) / (wi + wj), wi + wj]
            clusters.pop(j)
            merged = True
    total = sum(c[1] for c in clusters)
    clusters = [c for c in clusters if c[1] / total >= MIN_SHARE]
    total = sum(c[1] for c in clusters) or 1.0
    palette = []
    for c, n in sorted(clusters, key=lambda x: -x[1]):
        name, de = nearest_colour(c)
        palette.append({"lab": [round(float(v), 2) for v in c], "weight": round(float(n / total), 4),
                        "name": name, "delta_e": round(de, 1)})
    return palette


def is_multicolour(palette: list[dict]) -> bool:
    """Three or more clearly different colours, each covering a real share of the garment."""
    return len([p for p in palette if p["weight"] >= 0.15]) >= 3


def palette_similarity(a: list[dict], b: list[dict], sigma: float = 14.0) -> float:
    """Symmetric soft match between two palettes in [0, 1].

    Each colour finds its closest counterpart in the other palette; the CIEDE2000 distance is
    weighted by how much of the garment that colour covers, averaged both ways and mapped through
    exp(-(d / sigma)^2): ~1 for the same colours, ~0.6 for neighbouring shades (dE00 ~ 10),
    near 0 for different colours. Works on palettes stored by older versions too (only 'lab'
    and 'weight' are used).
    """
    if not a or not b:
        return float("nan")
    la = np.array([p["lab"] for p in a]); wa = np.array([p["weight"] for p in a])
    lb = np.array([p["lab"] for p in b]); wb = np.array([p["weight"] for p in b])
    d = ciede2000(la[:, None, :], lb[None, :, :])
    d_ab = float((d.min(axis=1) * wa).sum() / max(1e-9, wa.sum()))
    d_ba = float((d.min(axis=0) * wb).sum() / max(1e-9, wb.sum()))
    return float(np.exp(-((0.5 * (d_ab + d_ba)) / sigma) ** 2))


SHADE_SHIFT = {"dark": -18.0, "darker": -18.0, "deep": -14.0, "light": 18.0, "lighter": 18.0, "pastel": 22.0}


def requested_colour_lab(name: str, shade: str | None = None) -> np.ndarray | None:
    """LAB target for a requested colour, optionally shifted lighter/darker ('darker blue')."""
    lab = colour_name_lab(name)
    if lab is None:
        return None
    lab = lab.copy()
    if shade in SHADE_SHIFT:
        lab[0] = float(np.clip(lab[0] + SHADE_SHIFT[shade], 5, 97))
        if shade == "pastel":
            lab[1:] *= 0.5
    return lab


def lab_similarity(a: np.ndarray | None, b: np.ndarray | None, sigma: float = 15.0) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.exp(-(float(ciede2000(a, b)) / sigma) ** 2))


def name_similarity(a: str | None, b: str | None, sigma: float = 15.0) -> float:
    """Similarity of two colour *names* (e.g. metadata 'Navy Blue' vs requested 'blue'), CIEDE2000-based."""
    la, lb = (colour_name_lab(a) if a else None), (colour_name_lab(b) if b else None)
    return lab_similarity(la, lb, sigma)


# Colour families as shoppers use the words. A product colour may belong to two families (teal is a
# blue-green, olive a yellowish green, mauve a pinkish purple...).
COLOUR_FAMILIES = {
    "green": {"green", "bottle green", "lime green", "mint", "sea green", "olive", "teal"},
    "blue": {"blue", "navy blue", "royal blue", "sky blue", "turquoise blue", "teal"},
    "red": {"red", "maroon", "burgundy", "wine", "rust", "coral"},
    "pink": {"pink", "rose", "magenta", "mauve", "peach", "coral"},
    "purple": {"purple", "lavender", "mauve", "magenta", "wine"},
    "yellow": {"yellow", "mustard", "gold", "lime green"},
    "orange": {"orange", "rust", "coral", "peach", "mustard"},
    "brown": {"brown", "coffee brown", "tan", "khaki", "rust", "beige"},
    "white": {"white", "off white", "cream"},
    "black": {"black", "charcoal"},
    "grey": {"grey", "silver", "charcoal"},
    "beige": {"beige", "cream", "khaki", "tan", "off white"},
    "gold": {"gold", "mustard", "yellow"},
    "silver": {"silver", "grey"},
}


def colour_family(name: str | None) -> set[str]:
    """Product colour names that satisfy a shopper asking for `name`. A family word ('green') -> the family.
    A specific shade ('bottle green', 'navy blue') -> every family it belongs to (ranking then puts the closest
    shade first). A colour in no family -> itself plus names within CIEDE2000 12."""
    n = normalise_colour_name(name)
    if n is None:
        return set()
    if n in COLOUR_FAMILIES:
        return set(COLOUR_FAMILIES[n])
    fams = [f for f in COLOUR_FAMILIES.values() if n in f]
    if fams:
        return set().union(*fams)
    ref = _REF_LAB[_NAMES.index(n)]
    return {n} | {m for m in _NAMES if float(ciede2000(ref, _REF_LAB[_NAMES.index(m)])) < 12}


def colour_family_match(requested: str | None, product_colour: str | None) -> float:
    """1.0 if the product colour is in the requested colour's family, 0.0 if not, NaN if either is unknown."""
    fam = colour_family(requested)
    pc = normalise_colour_name(product_colour)
    if not fam or pc is None:
        return float("nan")
    return 1.0 if pc in fam else 0.0
