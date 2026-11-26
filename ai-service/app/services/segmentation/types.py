"""Structured results returned by every segmentation method (and by the pipeline)."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# garment groups shared by all backends (detector / parser labels are mapped onto these)
GROUPS = ("upper", "lower", "full_body", "outer", "accessory", "garment")


@dataclass
class GarmentInstance:
    label: str                    # model label, e.g. "long sleeve dress", "Upper-clothes", "garment region"
    group: str                    # one of GROUPS
    confidence: float             # 0..1, method-specific, never inflated
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in image pixels (x1/y1 exclusive)
    mask: np.ndarray | None       # bool HxW at image resolution (None for box-only detections)
    source: str                   # which method produced it
    occlusion: float = 0.0        # share of the garment outline hidden by other objects (estimate)
    alpha: np.ndarray | None = None  # optional soft mask 0..1 (boundary refinement / transparency)

    @property
    def area(self) -> int:
        if self.mask is not None:
            return int(self.mask.sum())
        x0, y0, x1, y1 = self.bbox
        return max(0, x1 - x0) * max(0, y1 - y0)

    def summary(self, image_area: int) -> dict:
        return {"label": self.label, "group": self.group, "confidence": round(float(self.confidence), 3),
                "bbox": [int(v) for v in self.bbox], "area_share": round(self.area / max(1, image_area), 4),
                "occlusion": round(float(self.occlusion), 3), "source": self.source,
                "has_mask": self.mask is not None}


@dataclass
class SegmentationResult:
    path: str                     # product_only | person | multi_garment | difficult | disabled
    method: str                   # final method actually used
    confidence: float
    image_size: tuple[int, int]   # (w, h)
    instances: list[GarmentInstance] = field(default_factory=list)
    target_index: int | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    attempts: list[dict] = field(default_factory=list)   # every method tried: {method, ok, reason, ms}
    timings_ms: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    scene: dict = field(default_factory=dict)
    cached: bool = False

    @property
    def target(self) -> GarmentInstance | None:
        if self.target_index is None or not self.instances:
            return None
        return self.instances[self.target_index]

    @property
    def mask(self) -> np.ndarray | None:
        t = self.target
        return None if t is None else t.mask

    @property
    def bbox(self):
        t = self.target
        return None if t is None else t.bbox

    def to_dict(self, include_masks: bool = False) -> dict:
        area = self.image_size[0] * self.image_size[1]
        out = {"path": self.path, "method": self.method, "confidence": round(float(self.confidence), 3),
               "image_size": list(self.image_size), "target_index": self.target_index,
               "bbox": list(self.bbox) if self.bbox else None,
               "fallback_used": self.fallback_used, "fallback_reason": self.fallback_reason,
               "instances": [i.summary(area) for i in self.instances], "attempts": self.attempts,
               "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
               "quality": self.quality, "scene": self.scene, "cached": self.cached}
        out = plain(out)
        if include_masks:
            for d, inst in zip(out["instances"], self.instances):
                d["mask_png_base64"] = mask_to_png_b64(inst.mask) if inst.mask is not None else None
        return out


def plain(v):
    """numpy scalars / tuples -> JSON-safe Python values (FastAPI cannot encode np.float64)."""
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def mask_to_png_b64(mask: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def valid_bbox(b, w: int, h: int) -> bool:
    return b is not None and 0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h
