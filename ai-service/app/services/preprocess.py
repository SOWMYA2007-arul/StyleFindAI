"""Step 1-2 of the pipeline: validate an uploaded file and turn it into a clean RGB image.

Model-specific resizing / normalisation (e.g. CLIP's 224x224 + mean/std) is done by each
encoder's own processor, because using the wrong normalisation silently ruins embeddings.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MIN_SIDE = 48          # Kaggle "small" catalogue images are 60x80, so stay below that
MAX_SIDE = 8000        # guards against decompression bombs
MAX_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageValidationError(ValueError):
    """Raised for user-facing validation problems (safe to show the message)."""


@dataclass
class LoadedImage:
    image: Image.Image          # RGB, EXIF-rotated, size-capped
    original_format: str
    width: int
    height: int
    sha256: str
    quality_warnings: list[str]


def load_and_validate(data: bytes, max_bytes: int = 8 * 1024 * 1024) -> LoadedImage:
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ImageValidationError(f"Image is larger than {max_bytes // (1024 * 1024)} MB.")

    # Pillow checks the real file signature, not the extension the user typed.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, ImportError, Image.DecompressionBombError):
        # ImportError: some optional packages (e.g. ultralytics) patch Image.open and try to import a HEIF
        # plugin for unknown files - still just "not a valid image" for the user.
        raise ImageValidationError("The file is not a valid image.") from None
    if fmt not in ALLOWED_FORMATS:
        raise ImageValidationError("Only JPG, JPEG, PNG and WEBP images are supported.")

    # verify() leaves the image unusable, so reopen to decode.
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # respect phone camera rotation
        img = to_rgb(img)
    except (OSError, Image.DecompressionBombError):
        raise ImageValidationError("The image could not be decoded.") from None

    w, h = img.size
    if min(w, h) < MIN_SIDE:
        raise ImageValidationError(f"Image is too small ({w}x{h}); minimum side is {MIN_SIDE}px.")
    if max(w, h) > MAX_SIDE:
        raise ImageValidationError(f"Image is too large ({w}x{h}); maximum side is {MAX_SIDE}px.")

    # Downscale very large photos once; every encoder works at <= 518px anyway.
    if max(w, h) > 1600:
        img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

    return LoadedImage(
        image=img,
        original_format=fmt,
        width=w,
        height=h,
        sha256=hashlib.sha256(data).hexdigest(),
        quality_warnings=quality_checks(img),
    )


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any mode to RGB. Transparent pixels become white (like catalogue photos)."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def quality_checks(img: Image.Image) -> list[str]:
    """Cheap heuristics that explain poor results to the user; they never block a search."""
    warnings: list[str] = []
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    mean = float(gray.mean())
    if mean < 40:
        warnings.append("Image is very dark; colour matching may be unreliable.")
    elif mean > 235:
        warnings.append("Image is very bright / washed out.")
    # Variance of the Laplacian is a standard blur measure.
    try:
        import cv2

        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        if lap_var < 15:
            warnings.append("Image looks blurry; pattern matching may be unreliable.")
    except ImportError:  # pragma: no cover
        pass
    w, h = img.size
    if max(w, h) / max(1, min(w, h)) > 4:
        warnings.append("Unusual aspect ratio; consider cropping to the garment.")
    return warnings
