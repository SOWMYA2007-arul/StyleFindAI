"""Garment segmentation pipeline tests.

Weight-free paths run for real. Optional models (clothing YOLO, SegFormer, SAM 2, pose) are exercised
with injected fakes that follow the real interfaces, plus "unavailable" cases with missing weights.
A smoke test with real Ultralytics weights runs only if yolo11n-seg.pt is present locally.
"""
from __future__ import annotations

import base64
import io
import random
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.config import get_settings
from app.main import app
from app.services.segmentation import backends as B
from app.services.segmentation import pipeline as P
from app.services.segmentation.refine import boundary_f1, dice, iou, precision_recall
from app.services.segmentation.types import GarmentInstance, bbox_of, valid_bbox

import make_segmentation_fixture as F  # tests/conftest.py puts scripts/ on sys.path

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    s = get_settings()
    for k, v in {"segmentation_strategy": "auto", "garment_detector": "none", "person_segmenter": "none",
                 "human_parser": "none", "enable_sam2": False, "pose_estimator": "none", "garment_isolation": "auto",
                 "enable_garment_detection": False, "min_segmentation_confidence": 0.4,
                 "segmentation_timeout_s": 20.0}.items():
        monkeypatch.setattr(s, k, v)
    P.cache().clear()
    for k in ("garments", "person", "parser", "sam2", "pose"):
        B.set_backend(k, None)
    yield
    P.cache().clear()
    for k in ("garments", "person", "parser", "sam2", "pose"):
        B.set_backend(k, None)


def scene(cat: str, seed: int = 3):
    img, insts, _ = F.scene(cat, random.Random(seed))
    union = np.zeros((img.size[1], img.size[0]), bool)
    for _, m in insts:
        union |= np.asarray(m) > 127
    return img, union, [(lab, np.asarray(m) > 127) for lab, m in insts]


def union_of(res):
    m = np.zeros((res.image_size[1], res.image_size[0]), bool)
    for g in res.instances:
        if g.mask is not None:
            m |= g.mask
    return m


class Exploding(B._Lazy):
    """Fails loudly if the pipeline calls it (used to prove a path does NOT run a model)."""
    name = "exploding"

    def instances(self, *a, **k):
        raise AssertionError("expensive model must not run on this path")

    parse = landmarks = refine = instances


class FakeDetector(B._Lazy):
    name = "garment_detector"

    def __init__(self, items):
        super().__init__()
        self.items, self.calls = items, 0

    def instances(self, img, want="garments"):
        self.calls += 1
        return [GarmentInstance(lab, B.label_group(lab) or "garment", c, bbox_of(m), m, "fake-yolo") for lab, c, m in self.items]


# ------------------------------------------------------------------ routing
def test_product_photo_uses_light_path_without_models():
    s = get_settings()
    s.garment_detector, s.enable_sam2, s.human_parser = "yolo", True, "segformer"
    for k in ("garments", "sam2", "parser"):
        B.set_backend(k, Exploding())
    img, gt, _ = scene("product_only")
    res = P.segment(img)
    assert res.path == "product_only" and res.method == "foreground"
    assert not res.fallback_used
    assert iou(union_of(res), gt) > 0.95
    assert res.attempts == []                      # no optional model was tried


def test_person_photo_removes_skin_and_face():
    img, gt, _ = scene("person_wearing", seed=0)
    res = P.segment(img, strategy="light")
    assert res.path == "person"
    m = union_of(res)
    assert iou(m, gt) > 0.85
    p, _ = precision_recall(m, gt)
    assert p > 0.9                                 # head / arms are not part of the garment mask


def test_multi_garment_instances_and_target_selection():
    img, gt, insts = scene("multi_garment", seed=0)
    res = P.segment(img, strategy="light")
    assert len(res.instances) >= 2
    lower = P.segment(img, target="lower", strategy="light")
    assert lower.target is not None and lower.target.group == "lower"
    gt_lower = next(m for lab, m in insts if lab == "trousers")
    assert iou(lower.target.mask, gt_lower) > 0.7
    by_index = P.segment(img, target=0, strategy="light")
    assert by_index.target_index == 0


def test_unknown_target_falls_back_to_prominent_garment():
    img, _, _ = scene("product_only")
    res = P.segment(img, target="lehenga")
    assert res.target is not None and "not found" in res.fallback_reason


# ------------------------------------------------------------------ optional models unavailable
def test_yolo_enabled_but_weights_missing_falls_back(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "garment_detector", "yolo")
    monkeypatch.setattr(s, "yolo_weights", str(Path("definitely") / "missing-weights.pt"))
    img, gt, _ = scene("person_wearing", seed=0)
    res = P.segment(img)
    att = [a for a in res.attempts if a["method"] == "garment_detector"]
    assert att and att[0]["ok"] is False and "unavailable" in att[0]["reason"]
    assert res.target is not None and iou(union_of(res), gt) > 0.8


def test_yolo_disabled_is_never_loaded():
    img, _, _ = scene("multi_garment", seed=1)
    P.segment(img)
    assert "garments" not in B._backends


def test_sam2_enabled_without_checkpoint_falls_back(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "enable_sam2", True)
    monkeypatch.setattr(s, "sam2_checkpoint", "")
    img, _, _ = scene("person_wearing", seed=0)
    res = P.segment(img)
    att = [a for a in res.attempts if a["method"] == "sam2"]
    assert att and not att[0]["ok"] and "SAM2_CHECKPOINT" in att[0]["reason"]
    assert res.target is not None


def test_parser_unavailable_falls_back(monkeypatch):
    class Broken(B._Lazy):
        name = "human_parser"

        def _load(self):
            raise OSError("model files not found")

        def instances(self, img):
            self.require()

    monkeypatch.setattr(get_settings(), "human_parser", "segformer")
    B.set_backend("parser", Broken())
    img, gt, _ = scene("person_wearing", seed=0)
    res = P.segment(img)
    att = [a for a in res.attempts if a["method"] == "human_parser"]
    assert att and "model files not found" in att[0]["reason"]
    assert res.method == "classical_person" and iou(union_of(res), gt) > 0.8


def test_slow_model_times_out(monkeypatch):
    class Slow(FakeDetector):
        def instances(self, img, want="garments"):
            time.sleep(2)
            return []

    s = get_settings()
    monkeypatch.setattr(s, "garment_detector", "yolo")
    monkeypatch.setattr(s, "segmentation_timeout_s", 0.3)
    B.set_backend("garments", Slow([]))
    img, _, _ = scene("person_wearing", seed=0)
    t = time.perf_counter()
    res = P.segment(img)
    assert time.perf_counter() - t < 1.8
    att = [a for a in res.attempts if a["method"] == "garment_detector"][0]
    assert not att["ok"] and "Timeout" in att["reason"]


# ------------------------------------------------------------------ optional models (fakes)
def test_clothing_detector_instances_are_used(monkeypatch):
    img, _, insts = scene("multi_garment", seed=0)
    monkeypatch.setattr(get_settings(), "garment_detector", "yolo")
    det = FakeDetector([("short_sleeve_top" if lab == "kurti" else lab, 0.9, m) for lab, m in insts])
    B.set_backend("garments", det)
    res = P.segment(img)
    assert det.calls == 1 and res.method.startswith("garment_detector")
    assert {g.group for g in res.instances} >= {"upper", "lower"}
    assert all(valid_bbox(g.bbox, *img.size) for g in res.instances)
    assert P.segment(img, target="trousers").target.label == "trousers"


def test_label_groups():
    assert B.label_group("short_sleeve_top") == "upper"
    assert B.label_group("trousers") == "lower"
    assert B.label_group("long sleeve dress") == "full_body"
    assert B.label_group("person") is None and B.label_group("face") is None and B.label_group("bag") is None


def test_segformer_parser_code_path_with_injected_model(monkeypatch):
    """Real SegformerParser logic with a tiny randomly-initialised SegFormer (ATR-style labels) - checks
    shapes / label mapping / body mask, not accuracy."""
    pytest.importorskip("transformers")
    from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor

    labels = ["Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt",
              "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"]
    cfg = SegformerConfig(num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)},
                          hidden_sizes=[8, 16, 24, 32], decoder_hidden_size=16, depths=[1, 1, 1, 1], num_attention_heads=[1, 1, 1, 1])
    parser = B.SegformerParser("test/random", SegformerForSemanticSegmentation(cfg), SegformerImageProcessor(size={"height": 64, "width": 64}))
    img, _, _ = scene("person_wearing", seed=0)
    out = parser.parse(img)
    assert out["labels"].shape == (img.size[1], img.size[0])
    inst, body = parser.instances(img)
    assert body.shape == out["labels"].shape
    assert all(g.group in ("upper", "lower", "full_body", "outer", "accessory", "garment", "footwear") for g in inst)


def test_sam2_candidate_scoring_with_fake_predictor(monkeypatch):
    img, gt, _ = scene("person_wearing", seed=5)
    h, w = gt.shape

    class Pred:
        def __init__(self):
            self.set_calls = 0

        def set_image(self, rgb):
            self.set_calls += 1

        def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
            whole = np.ones((h, w), bool)            # high score but covers everything
            exact = gt.copy()                        # the right answer, lower raw score
            return np.stack([whole, exact]), np.array([0.95, 0.80]), None

    pred = Pred()
    B.set_backend("sam2", B.Sam2Refiner(predictor=pred))
    s = get_settings()
    monkeypatch.setattr(s, "enable_sam2", True)
    res = P.segment(img, image_key="abc")
    if res.target is not None and "sam2" in res.method:
        assert iou(res.target.mask, gt) > 0.95       # consistency + box terms beat the raw SAM score
    m, info = B.select_mask(np.stack([np.ones((h, w), bool), gt]), np.array([0.95, 0.8]), bbox_of(gt), gt, None)
    assert (m == gt).all() and len(info["candidates"]) == 2


def test_sam2_never_runs_on_product_photos(monkeypatch):
    monkeypatch.setattr(get_settings(), "enable_sam2", True)
    B.set_backend("sam2", Exploding())
    img, _, _ = scene("product_only")
    res = P.segment(img)
    assert res.path == "product_only" and not any(a["method"] == "sam2" for a in res.attempts)


def test_pose_only_on_person_path(monkeypatch):
    monkeypatch.setattr(get_settings(), "pose_estimator", "mediapipe")
    B.set_backend("pose", Exploding())
    img, _, _ = scene("patterned")
    assert P.segment(img).path == "product_only"


# ------------------------------------------------------------------ fallback / confidence / validity
def test_low_confidence_falls_back_to_full_image(monkeypatch):
    monkeypatch.setattr(get_settings(), "min_segmentation_confidence", 0.99)
    img, _, _ = scene("person_wearing", seed=0)
    res = P.segment(img)
    assert res.fallback_used and res.target_index is None and "below" in res.fallback_reason
    crop, info = P.isolate(img)
    assert crop.size == img.size and info["fallback_used"]


def test_masks_and_boxes_are_valid_for_every_category():
    for cat in F.CATEGORIES:
        img, _, _ = scene(cat, seed=7)
        res = P.segment(img, strategy="light")
        for g in res.instances:
            assert g.mask is not None and g.mask.shape == (img.size[1], img.size[0]) and g.mask.dtype == bool
            assert valid_bbox(g.bbox, *img.size)
            assert 0.0 <= g.confidence <= 1.0
        assert 0.0 <= res.confidence <= 1.0


def test_thin_details_are_preserved():
    img, gt, _ = scene("thin_details", seed=2)
    res = P.segment(img)
    m = union_of(res)
    assert boundary_f1(m, gt) > 0.6
    straps = gt.copy(); straps[64:, :] = False      # strap pixels only (above the bodice)
    if straps.sum() > 30:
        assert (m & straps).sum() / straps.sum() > 0.5


def test_occluding_hand_is_not_filled_in():
    # (a hand overlapping the garment's outline is not removed - known limitation, see docs/SEGMENTATION.md)
    img, gt, _ = scene("occluded", seed=0)
    res = P.segment(img)
    assert iou(union_of(res), gt) > 0.95
    assert res.scene.get("occluded_share", 0) > 0.05
    assert res.target.occlusion > 0.05 and res.confidence < 0.85    # confidence drops, pixels are not invented


def test_isolation_crop_is_white_outside_garment():
    img, gt, _ = scene("product_only")
    crop, info = P.isolate(img)
    assert crop.size[0] < img.size[0] and info["crop_box"]
    a = np.asarray(crop)
    assert (a[0, 0] >= 250).all() and (a[-1, -1] >= 250).all()


def test_invalid_input_rejected():
    with pytest.raises(Exception):
        P.segment("not an image")  # type: ignore[arg-type]
    r = client.post("/ai/segment", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422
    good = io.BytesIO(); scene("product_only")[0].save(good, "PNG")
    r = client.post("/ai/segment", files={"image": ("x.png", good.getvalue(), "image/png")}, data={"strategy": "magic"})
    assert r.status_code == 422
    r = client.post("/ai/segment", files={"image": ("x.png", good.getvalue(), "image/png")}, data={"target": "<script>"})
    assert r.status_code == 422


# ------------------------------------------------------------------ timings / cache
def test_timings_logged_per_stage():
    img, _, _ = scene("person_wearing", seed=0)
    res = P.segment(img, strategy="light")
    for k in ("foreground", "quality", "scene", "classical_person", "total"):
        assert k in res.timings_ms and res.timings_ms[k] >= 0
    assert res.quality["width"] == img.size[0]


def test_cache_reuse_by_image_key():
    img, _, _ = scene("person_wearing", seed=0)
    a = P.segment(img, image_key="sha-1")
    t = time.perf_counter()
    b = P.segment(img, image_key="sha-1")
    assert b.cached and (time.perf_counter() - t) < 0.01
    c = P.segment(img, image_key="sha-1", target="lower")      # different target -> different entry
    assert not c.cached or c is not a


def test_cache_key_changes_with_configuration(monkeypatch):
    img, _, _ = scene("product_only")
    P.segment(img, image_key="sha-2")
    monkeypatch.setattr(get_settings(), "min_segmentation_confidence", 0.5)
    assert not P.segment(img, image_key="sha-2").cached


# ------------------------------------------------------------------ API + search integration
def test_segment_endpoint_returns_instances_and_masks():
    img, _, _ = scene("multi_garment", seed=0)
    buf = io.BytesIO(); img.save(buf, "PNG")
    r = client.post("/ai/segment", files={"image": ("m.png", buf.getvalue(), "image/png")},
                    data={"include_masks": "true", "strategy": "light"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["path"] in ("person", "multi_garment", "difficult") and d["instances"]
    m = Image.open(io.BytesIO(base64.b64decode(d["instances"][0]["mask_png_base64"])))
    assert m.size == img.size
    assert "total" in d["timings_ms"]


def test_segmentation_status_endpoint():
    d = client.get("/ai/segmentation/status").json()
    assert d["backends"]["garment_detector"] == "none" and "mask_cache" in d


def test_detection_isolate_garment_modes(monkeypatch):
    from app.services.detection import isolate_garment

    img, _, _ = scene("product_only")
    s = get_settings()
    monkeypatch.setattr(s, "garment_isolation", "off")
    out, info = isolate_garment(img)
    assert out is img and info["method"] == "full_image"
    monkeypatch.setattr(s, "garment_isolation", "foreground")
    out, info = isolate_garment(img)
    assert info["method"] == "foreground_crop" and info["segmentation"]["path"] == "product_only"
    monkeypatch.setattr(s, "garment_isolation", "auto")
    out, info = isolate_garment(img, "k", None)
    assert info["method"] == "foreground_crop"


def test_query_embedding_cache(monkeypatch):
    from app.models.encoders import get_registry
    from app.services import search
    from app.services.preprocess import load_and_validate

    img, _, _ = scene("product_only")
    buf = io.BytesIO(); img.save(buf, "PNG")
    loaded = load_and_validate(buf.getvalue())
    enc = get_registry().get("cv-baseline")
    calls = {"n": 0}
    orig = enc.embed_images

    def counting(images):
        calls["n"] += 1
        return orig(images)

    monkeypatch.setattr(enc, "embed_images", counting)
    search.embedding_cache().clear()
    a = search.prepare_query(loaded, "cv-baseline", crop=True)
    b = search.prepare_query(loaded, "cv-baseline", crop=True)
    assert calls["n"] == 1 and b.timings.get("embedding_cache_hit") == 1
    assert np.allclose(a.embedding, b.embedding)


# ------------------------------------------------------------------ metrics
def test_metrics_basic_values():
    a = np.zeros((50, 50), bool); a[10:30, 10:30] = True
    b = np.zeros((50, 50), bool); b[20:40, 10:30] = True
    assert iou(a, a) == 1.0 and dice(a, a) == 1.0 and boundary_f1(a, a) == 1.0
    assert abs(iou(a, b) - 200 / 600) < 1e-6 and abs(dice(a, b) - 0.5) < 1e-6
    assert precision_recall(a, b) == (0.5, 0.5)


# ------------------------------------------------------------------ real weights (only if present locally)
WEIGHTS = [p for p in (Path("yolo11n-seg.pt"), Path(get_settings().model_cache_path) / "yolo" / "yolo11n-seg.pt") if p.exists()]


@pytest.mark.skipif(not WEIGHTS, reason="yolo11n-seg.pt not downloaded (optional)")
def test_real_ultralytics_person_segmenter_smoke(monkeypatch):
    """COCO weights: checks the real person-mask code path runs. COCO has no garment classes, so the
    garment detector must return nothing with these weights (no false 'garment' claims)."""
    pytest.importorskip("ultralytics")
    img, _, _ = scene("person_wearing", seed=0)
    person = B.YoloBackend(str(WEIGHTS[0]), "person_segmenter")
    out = person.instances(img, "person")  # drawings may not be detected; must not crash
    assert isinstance(out, list)
    garments = B.YoloBackend(str(WEIGHTS[0]), "garment_detector").instances(img, "garments")
    assert garments == []
