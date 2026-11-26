"""Cheap image-quality assessment that runs BEFORE any expensive model.

Nothing here rejects a photo that can still be searched; it explains likely problems and tells the
pipeline when advanced segmentation would be wasted effort.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def assess(img: Image.Image, fg_share: float | None = None, exif_rotated: bool = False) -> dict:
    w, h = img.size
    gray = np.asarray(img.convert("L").resize((min(w, 512), max(1, int(h * min(w, 512) / w)))), dtype=np.float32)
    lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    mean, std = float(gray.mean()), float(gray.std())
    issues, flags = [], {}
    if min(w, h) < 224:
        issues.append(f"Low resolution ({w}x{h}); details such as embroidery may be lost.")
        flags["low_resolution"] = True
    if lap < 15:
        issues.append("The photo looks blurry; pattern matching may be unreliable.")
        flags["blurry"] = True
    if mean < 40:
        issues.append("The photo is very dark.")
        flags["dark"] = True
    elif mean > 235:
        issues.append("The photo is very bright / washed out.")
        flags["bright"] = True
    if std < 18:
        issues.append("Very low contrast; the garment may blend into the background.")
        flags["low_contrast"] = True
    if fg_share is not None and fg_share < 0.08:
        issues.append("The garment fills only a small part of the photo; crop closer to it.")
        flags["small_garment"] = True
    return {"width": w, "height": h, "sharpness": round(lap, 1), "brightness": round(mean, 1),
            "contrast": round(std, 1), "garment_share": None if fg_share is None else round(float(fg_share), 3),
            "exif_rotated": exif_rotated, "flags": flags, "messages": issues,
            # usable = searchable at all; advanced models won't rescue a tiny, blurry or black image
            "advanced_worthwhile": not (flags.get("blurry") and flags.get("low_resolution")) and not flags.get("dark")}
