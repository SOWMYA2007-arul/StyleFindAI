"""Optional model backends. Every backend:
  * is loaded lazily on first real use and then reused,
  * reports `available()` without raising (missing package / weights -> unavailable + reason),
  * is called through `run_with_timeout` by the pipeline, so a slow or broken model falls back
    instead of breaking search.

| backend            | task                                   | default weights                         | licence (verify!)                  |
|--------------------|----------------------------------------|-----------------------------------------|------------------------------------|
| YoloGarments       | clothing detection / instance segm.    | YOLO_WEIGHTS (clothing-trained, yours)  | Ultralytics AGPL-3.0 + weights' own |
| YoloPerson         | person instance masks (COCO 'person')  | yolo11n-seg.pt (auto-download, 6 MB)    | AGPL-3.0                           |
| SegformerParser    | human parsing (face/hair/arms/clothes) | mattmdjaga/segformer_b2_clothes (~110MB)| model card: MIT; data: ATR (research) |
| Sam2Refiner        | box/point-prompted mask refinement     | SAM2_CHECKPOINT (sam2.1_hiera_tiny.pt)  | Apache-2.0                         |
| MediaPipePose      | 33 body landmarks                      | POSE_MODEL_PATH (pose_landmarker_lite)  | Apache-2.0                         |

COCO-trained YOLO weights have NO garment classes - they are only used as a person segmenter.
"""
from __future__ import annotations

import concurrent.futures as cf
import re
import threading
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from app.config import get_settings
from app.services.segmentation.types import GarmentInstance, bbox_of
from app.utils.logging import get_logger

log = get_logger("segmentation.backends")

_executor = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="seg-model")


class ModelTimeout(RuntimeError):
    pass


def run_with_timeout(fn: Callable, timeout_s: float, *args, **kwargs):
    """Run an optional model call with a time limit. Python cannot kill the thread, but the request
    stops waiting and uses the next fallback (the late result is discarded)."""
    fut = _executor.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout_s)
    except cf.TimeoutError:
        raise ModelTimeout(f"model call exceeded {timeout_s:.0f}s") from None


# ------------------------------------------------------------------ garment groups
GROUP_KEYWORDS = [
    ("full_body", ("dress", "jumpsuit", "gown", "saree", "sari", "lehenga", "romper", "playsuit")),
    ("outer", ("outwear", "outerwear", "jacket", "coat", "blazer", "cardigan", "shrug", "cape", "poncho")),
    # upper BEFORE lower: DeepFashion2's "short_sleeve_top" must not match "short" (shorts)
    ("upper", ("top", "shirt", "blouse", "vest", "sling", "sweater", "hoodie", "kurta", "kurti", "tee", "upper", "tunic")),
    ("lower", ("trouser", "pant", "skirt", "short", "jean", "legging", "palazzo", "salwar", "churidar")),
    ("accessory", ("dupatta", "scarf", "shawl", "stole")),
]
NOT_GARMENT = ("shoe", "boot", "sandal", "sock", "bag", "hat", "cap", "glass", "glove", "belt", "watch", "jewel")
BODY = ("face", "hair", "skin", "arm", "leg", "neck", "head", "hand", "foot", "torso", "body")


def label_group(name: str) -> str | None:
    n = (name or "").lower().replace("_", " ").replace("-", " ")
    tokens = re.split(r"[^a-z]+", n)
    # whole words only: "cap" must not reject Fashionpedia's "cape"
    if any(t in (w, w + "s", w + "es") or (w == "jewel" and t.startswith(w)) for t in tokens for w in NOT_GARMENT) \
            or n in ("person", "background", "bg"):
        return None
    for g, words in GROUP_KEYWORDS:
        if any(w in n for w in words):
            return g
    if any(w in n for w in BODY):
        return None
    return "garment" if "cloth" in n else None


def _rasterize(polys, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        if p is not None and len(p) >= 3:
            cv2.fillPoly(m, [np.asarray(p, dtype=np.int32).reshape(-1, 1, 2)], 1)
    return m > 0


# ------------------------------------------------------------------ YOLO (ultralytics, optional)
class _Lazy:
    name = "backend"

    def __init__(self):
        self._model = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def _load(self):
        raise NotImplementedError

    def model(self):
        with self._lock:
            if self._model is None and self._error is None:
                try:
                    self._model = self._load()
                except Exception as e:  # noqa: BLE001 - optional component
                    self._error = f"{type(e).__name__}: {e}"[:300]
                    log.warning("%s unavailable: %s", self.name, self._error)
            return self._model

    def available(self) -> bool:
        return self.model() is not None

    def require(self):
        """The loaded model, or a clear error (recorded as the fallback reason by the pipeline)."""
        m = self.model()
        if m is None:
            raise RuntimeError(f"{self.name} unavailable ({self._error})")
        return m

    def status(self) -> dict:
        return {"loaded": self._model is not None, "error": self._error}


class YoloBackend(_Lazy):
    def __init__(self, weights: str, name: str):
        super().__init__()
        self.weights, self.name = weights, name

    def _load(self):
        from ultralytics import YOLO  # optional dependency (AGPL-3.0)

        return YOLO(self.weights)

    def predict(self, img: Image.Image):
        s = get_settings()
        res = self.require().predict(img.convert("RGB"), verbose=False, conf=s.detection_min_confidence,
                                   imgsz=s.yolo_imgsz, max_det=max(1, s.detection_max_instances * 3),
                                   device=s.ai_device if s.ai_device != "cpu" else "cpu")
        return res[0]

    def instances(self, img: Image.Image, want: str = "garments") -> list[GarmentInstance]:
        """want='garments' -> clothing classes only; want='person' -> COCO person masks/boxes."""
        s = get_settings()
        r = self.predict(img)
        w, h = img.size
        names = r.names
        polys = r.masks.xy if getattr(r, "masks", None) is not None else None
        out = []
        for i, b in enumerate(r.boxes):
            label = str(names[int(b.cls)])
            conf = float(b.conf)
            x0, y0, x1, y1 = [int(round(v)) for v in b.xyxy[0].tolist()]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) < s.detection_min_box_area * w * h:
                continue
            group = "person" if label.lower() == "person" else label_group(label)
            if (want == "person") != (group == "person") or group is None:
                continue
            mask = _rasterize([polys[i]], h, w) if polys is not None and i < len(polys) else None
            if mask is not None and not mask.any():
                mask = None
            out.append(GarmentInstance(label, group if group != "person" else "garment", conf,
                                       (x0, y0, x1, y1), mask, f"yolo:{self.weights}"))
        out.sort(key=lambda g: -g.confidence)
        return out[: s.detection_max_instances]


# ------------------------------------------------------------------ SegFormer human parsing (transformers, optional weights)
class SegformerParser(_Lazy):
    name = "human_parser"

    def __init__(self, model_id: str, model=None, processor=None):
        super().__init__()
        self.model_id = model_id
        if model is not None:  # injected (tests)
            self._model = (model.eval(), processor)

    def _load(self):
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        s = get_settings()
        cache = str(s.model_cache_path)
        proc = AutoImageProcessor.from_pretrained(self.model_id, cache_dir=cache)
        model = AutoModelForSemanticSegmentation.from_pretrained(self.model_id, cache_dir=cache).to(s.ai_device).eval()
        return model, proc

    def parse(self, img: Image.Image) -> dict:
        """Returns {'labels': HxW int, 'prob': HxW max-prob, 'id2label': {...}} at image resolution."""
        import torch

        model, proc = self.require()
        s = get_settings()
        size = s.human_parser_size
        small = img.convert("RGB").copy()
        small.thumbnail((size, size), Image.Resampling.BILINEAR)
        inputs = proc(images=small, return_tensors="pt")
        with torch.inference_mode():
            logits = model(pixel_values=inputs["pixel_values"].to(s.ai_device)).logits
        w, h = img.size
        up = torch.nn.functional.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)[0]
        prob = up.softmax(0)
        conf, lab = prob.max(0)
        return {"labels": lab.cpu().numpy(), "prob": conf.cpu().numpy(), "id2label": dict(model.config.id2label)}

    def instances(self, img: Image.Image) -> tuple[list[GarmentInstance], np.ndarray]:
        p = self.parse(img)
        labels, prob, id2label = p["labels"], p["prob"], p["id2label"]
        out, body = [], np.zeros(labels.shape, bool)
        for k, name in id2label.items():
            m = labels == int(k)
            if not m.any():
                continue
            n = str(name).lower()
            g = label_group(n)
            if g is None:
                if any(b in n for b in BODY):
                    body |= m
                continue
            if m.sum() < 0.002 * m.size:
                continue
            out.append(GarmentInstance(str(name), g, float(prob[m].mean()), bbox_of(m), m, f"segformer:{self.model_id}"))
        out.sort(key=lambda x: -x.area)
        return out, body


# ------------------------------------------------------------------ SAM 2 refinement (optional)
class Sam2Refiner(_Lazy):
    name = "sam2"

    def __init__(self, predictor=None):
        super().__init__()
        if predictor is not None:
            self._model = predictor
        self._img_key = None
        self._pred_lock = threading.Lock()

    def _load(self):
        s = get_settings()
        if not s.sam2_checkpoint:
            raise FileNotFoundError("SAM2_CHECKPOINT is not set")
        from sam2.build_sam import build_sam2  # optional dependency (Apache-2.0)
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        return SAM2ImagePredictor(build_sam2(s.sam2_config, s.sam2_checkpoint, device=s.ai_device))

    def refine(self, rgb: np.ndarray, box, coarse: np.ndarray | None = None,
               exclude: np.ndarray | None = None, image_key=None) -> tuple[np.ndarray | None, dict]:
        """Box prompt + positive points (inside the coarse garment) + negative points (on skin/face/
        background), several candidate masks, choose by: SAM score, consistency with the coarse mask,
        little overlap with excluded regions, sensible area inside the box."""
        pred = self.require()
        pts, lbl = _prompt_points(coarse, exclude, box)
        with self._pred_lock:  # the predictor holds one image embedding at a time
            if image_key is None or image_key != self._img_key:
                pred.set_image(rgb)
                self._img_key = image_key
            masks, scores, _ = pred.predict(point_coords=pts if len(pts) else None,
                                            point_labels=lbl if len(lbl) else None,
                                            box=np.asarray(box, dtype=np.float32)[None, :], multimask_output=True)
        return select_mask(np.asarray(masks).astype(bool), np.asarray(scores, dtype=float), box, coarse, exclude)


def _prompt_points(coarse, exclude, box, n_pos: int = 3, n_neg: int = 3):
    rng = np.random.default_rng(0)
    pts, lbl = [], []
    if coarse is not None and coarse.any():
        inner = cv2.erode(coarse.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        ys, xs = np.where(inner if inner.any() else coarse)
        for i in rng.choice(len(ys), min(n_pos, len(ys)), replace=False):
            pts.append((xs[i], ys[i])); lbl.append(1)
    if exclude is not None and exclude.any():
        x0, y0, x1, y1 = box
        ex = np.zeros_like(exclude); ex[y0:y1, x0:x1] = exclude[y0:y1, x0:x1]
        ys, xs = np.where(ex)
        if len(ys):
            for i in rng.choice(len(ys), min(n_neg, len(ys)), replace=False):
                pts.append((xs[i], ys[i])); lbl.append(0)
    return np.asarray(pts, dtype=np.float32), np.asarray(lbl, dtype=np.int32)


def select_mask(masks: np.ndarray, scores: np.ndarray, box, coarse, exclude) -> tuple[np.ndarray | None, dict]:
    x0, y0, x1, y1 = box
    box_area = max(1, (x1 - x0) * (y1 - y0))
    best, best_s, info = None, -1e9, []
    for m, sc in zip(masks, scores):
        area = float(m.sum())
        fill = area / box_area
        cons = 0.0
        if coarse is not None and coarse.any():
            cons = float((m & coarse).sum() / max(1.0, float((m | coarse).sum())))
        bad = float((m & exclude).sum() / max(1.0, area)) if exclude is not None else 0.0
        outside = float(m.sum() - m[y0:y1, x0:x1].sum()) / max(1.0, area)
        total = float(sc) + 0.8 * cons - 1.5 * bad - 1.0 * outside - (0.5 if not 0.15 <= fill <= 0.98 else 0.0)
        info.append({"sam_score": round(float(sc), 3), "consistency": round(cons, 3), "excluded_overlap": round(bad, 3),
                     "outside_box": round(outside, 3), "box_fill": round(fill, 3), "total": round(total, 3)})
        if total > best_s:
            best, best_s = m, total
    return best, {"candidates": info, "chosen_total": round(best_s, 3)}


# ------------------------------------------------------------------ MediaPipe pose (optional)
class MediaPipePose(_Lazy):
    name = "pose"
    LANDMARKS = {"l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14, "l_wrist": 15, "r_wrist": 16,
                 "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26}

    def _load(self):
        s = get_settings()
        if not s.pose_model_path:
            raise FileNotFoundError("POSE_MODEL_PATH is not set")
        import mediapipe as mp  # optional dependency (Apache-2.0)
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

        self._mp = mp
        return PoseLandmarker.create_from_options(PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=s.pose_model_path)))

    def landmarks(self, img: Image.Image) -> dict | None:
        mp = self._mp
        res = self.require().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img.convert("RGB"))))
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        w, h = img.size
        return {k: (lm[i].x * w, lm[i].y * h, getattr(lm[i], "visibility", 1.0)) for k, i in self.LANDMARKS.items()}


def limb_mask(lm: dict, shape: tuple[int, int], thickness: int) -> np.ndarray:
    """Thick lines along visible forearms/upper arms and lower legs: where exposed skin is expected."""
    m = np.zeros(shape, np.uint8)
    for a, b in (("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist")):
        pa, pb = lm.get(a), lm.get(b)
        if pa and pb and pa[2] > 0.5 and pb[2] > 0.5:
            cv2.line(m, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), 1, thickness)
    return m > 0


def hip_line(lm: dict) -> float | None:
    ys = [lm[k][1] for k in ("l_hip", "r_hip") if k in lm and lm[k][2] > 0.5]
    return float(np.mean(ys)) if ys else None


# ------------------------------------------------------------------ registry
_backends: dict[str, _Lazy] = {}


def get_backend(kind: str) -> _Lazy | None:
    """kind: garments | person | parser | sam2 | pose. None when not configured."""
    s = get_settings()
    configured = {
        "garments": s.garment_detector == "yolo" or s.enable_garment_detection or s.garment_isolation == "yolo",
        "person": s.person_segmenter == "yolo",
        "parser": s.human_parser == "segformer",
        "sam2": s.enable_sam2,
        "pose": s.pose_estimator == "mediapipe",
    }
    if not configured.get(kind):
        return None
    if kind not in _backends:
        _backends[kind] = {
            "garments": lambda: YoloBackend(s.yolo_weights, "garment_detector"),
            "person": lambda: YoloBackend(s.person_yolo_weights, "person_segmenter"),
            "parser": lambda: SegformerParser(s.human_parser_model),
            "sam2": Sam2Refiner,
            "pose": MediaPipePose,
        }[kind]()
    return _backends[kind]


def set_backend(kind: str, backend: _Lazy | None) -> None:
    """Inject / remove a backend (tests)."""
    if backend is None:
        _backends.pop(kind, None)
    else:
        _backends[kind] = backend


def status() -> dict:
    s = get_settings()
    return {"garment_detector": s.garment_detector, "person_segmenter": s.person_segmenter, "human_parser": s.human_parser,
            "sam2": s.enable_sam2, "pose": s.pose_estimator,
            "loaded": {k: b.status() for k, b in _backends.items()}}
