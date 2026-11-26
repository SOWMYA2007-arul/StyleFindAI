"""Tests for the v2 improvements (retrieval correctness, garment isolation, colour, pattern, text,
model management). Unit tests run without a database; the rest use the real DB if available."""
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services.colour import (COLOUR_REFERENCE, ciede2000, extract_palette, nearest_colour_name, palette_similarity,
                                 requested_colour_lab, rgb_to_lab)
from app.services.foreground import estimate_foreground, mask_iou
from app.services.pattern import classify_pattern
from app.utils.keywords import parse_text_query
from conftest import requires_db

POLY = [(110, 40), (190, 40), (230, 360), (70, 360)]


def garment(fill, bg=(246, 246, 246), draw=None, outline=None):
    img = Image.new("RGB", (300, 400), bg)
    d = ImageDraw.Draw(img)
    d.polygon(POLY, fill=fill)
    if outline:
        d.line(POLY + [POLY[0]], fill=outline, width=2)
    if draw:
        mask = Image.new("L", img.size, 0); ImageDraw.Draw(mask).polygon(POLY, fill=255)
        tex = img.copy(); draw(ImageDraw.Draw(tex))
        img.paste(tex, (0, 0), mask)
    return img


def truth_mask(size=(192, 256)):
    m = Image.new("L", (300, 400), 0); ImageDraw.Draw(m).polygon(POLY, fill=255)
    return np.asarray(m.resize(size)) > 127


# ------------------------------------------------------------------ colour (CIEDE2000)
def test_ciede2000_reference_value():
    # Sharma, Wu & Dalal (2005) test pair 1
    assert abs(float(ciede2000(np.array([50, 2.6772, -79.7751]), np.array([50, 0, -82.7485]))) - 2.0425) < 1e-3


@pytest.mark.parametrize("a,b", [("maroon", "red"), ("wine", "burgundy"), ("peach", "pink"), ("navy blue", "royal blue"),
                                 ("cream", "white"), ("mustard", "yellow"), ("olive", "green")])
def test_brief_colour_pairs_are_named_separately(a, b):
    lab = lambda n: rgb_to_lab(np.array([[COLOUR_REFERENCE[n]]], dtype=np.uint8))[0, 0]  # noqa: E731
    assert nearest_colour_name(lab(a)) == a and nearest_colour_name(lab(b)) == b


def test_palette_merges_shades_and_ignores_background():
    def shaded(d):
        for y in range(40, 360, 2):
            d.line([(0, y), (300, y)], fill=(200 - (y % 10), 30, 40))
    pal = extract_palette(garment((200, 30, 40), draw=shaded))
    assert pal[0]["name"] == "red" and pal[0]["weight"] > 0.8  # one colour, not "red + dark red + white"


def test_palette_similarity_and_shade_request():
    red, maroon = extract_palette(garment((205, 30, 40))), extract_palette(garment((110, 15, 30)))
    assert palette_similarity(red, red) > 0.95 > palette_similarity(red, maroon)
    dark = requested_colour_lab("blue", "dark"); base = requested_colour_lab("blue")
    assert dark[0] < base[0]


# ------------------------------------------------------------------ foreground / garment isolation
@pytest.mark.parametrize("kind,min_iou", [("white_on_white", 0.8), ("shadow", 0.9), ("dark_bg", 0.9)])
def test_foreground_hard_cases(kind, min_iou):
    if kind == "white_on_white":
        img = garment((238, 238, 236), bg=(246, 246, 244), outline=(215, 215, 212))
    elif kind == "shadow":
        img = Image.new("RGB", (300, 400), (235, 235, 235)); d = ImageDraw.Draw(img)
        d.polygon([(x + 25, y + 15) for x, y in POLY], fill=(180, 180, 180)); d.polygon(POLY, fill=(40, 90, 200))
    else:
        img = garment((180, 40, 60), bg=(25, 25, 30))
    assert mask_iou(estimate_foreground(img).mask, truth_mask()) >= min_iou


def test_foreground_close_up_fills_frame():
    # a close-up where the fabric touches every edge must not be treated as background
    img = Image.new("RGB", (300, 400), (25, 30, 80))
    d = ImageDraw.Draw(img)
    for x in range(0, 300, 30):
        d.ellipse((x, 100, x + 10, 110), fill=(250, 180, 200))
    fg = estimate_foreground(img)
    assert fg.mask.mean() > 0.8


def test_foreground_is_deterministic():
    img = garment((110, 50, 140), bg=(200, 210, 190))
    a, b = estimate_foreground(img, refine=True), estimate_foreground(img, refine=True)
    assert (a.mask == b.mask).all()


def test_detection_falls_back_without_yolo(monkeypatch):
    from app.config import get_settings
    from app.services import detection

    s = get_settings()
    monkeypatch.setattr(s, "garment_isolation", "yolo")
    monkeypatch.setattr(s, "yolo_weights", "definitely-missing-weights.pt")
    monkeypatch.setattr(detection, "_yolo", None)
    monkeypatch.setattr(detection, "_yolo_failed", False)
    out, info = detection.isolate_garment(garment((200, 30, 40)))
    assert info["method"] in ("foreground_crop", "full_image")
    assert "fallback_from" in info or info["method"] == "full_image"
    assert out.size[0] > 0


def test_foreground_crop_keeps_padding():
    from app.services.detection import foreground_isolation

    out, info = foreground_isolation(garment((200, 30, 40)))
    assert info["method"] == "foreground_crop"
    w = POLY[2][0] - POLY[3][0]
    assert out.size[0] > w  # wider than the garment itself: sleeves / edges are not cut off


# ------------------------------------------------------------------ pattern
def test_pattern_classifier_structural_patterns():
    stripes = garment((40, 90, 200), draw=lambda d: [d.rectangle((0, y, 300, y + 6), fill=(250, 250, 250)) for y in range(0, 400, 18)])
    checks = garment((40, 90, 200), draw=lambda d: ([d.rectangle((x, 0, x + 5, 400), fill=(250, 250, 250)) for x in range(0, 300, 18)],
                                                     [d.rectangle((0, y, 300, y + 5), fill=(250, 250, 250)) for y in range(0, 400, 18)]))
    dots = garment((40, 90, 200), draw=lambda d: [d.ellipse((x, y, x + 8, y + 8), fill=(250, 250, 250))
                                                  for y in range(0, 400, 22) for x in range(0, 300, 22)])
    assert classify_pattern(garment((40, 90, 200)))["value"] == "solid"
    assert classify_pattern(stripes)["value"] == "striped"
    assert classify_pattern(checks)["value"] == "checked"
    assert classify_pattern(dots)["value"] == "polka dot"


def test_pattern_classifier_says_unknown_instead_of_guessing():
    rng = np.random.default_rng(0)

    def blobs(d):
        for _ in range(120):
            x, y = rng.integers(0, 300), rng.integers(0, 400)
            d.polygon([(x, y), (x + rng.integers(5, 25), y + rng.integers(-10, 10)), (x + rng.integers(-10, 10), y + rng.integers(5, 25))],
                      fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    r = classify_pattern(garment((200, 200, 60), draw=blobs))
    assert r["value"] in ("unknown",)  # abstract print: left to the zero-shot model


def test_merge_pattern_prefers_the_right_source():
    from app.services.attributes import merge_pattern

    assert merge_pattern({"value": "striped", "confidence": 0.85}, {"value": "floral", "confidence": 0.45})["value"] == "striped"
    assert merge_pattern({"value": "unknown", "confidence": 0}, {"value": "paisley", "confidence": 0.7})["value"] == "paisley"
    agree = merge_pattern({"value": "checked", "confidence": 0.6}, {"value": "checked", "confidence": 0.6})
    assert agree["value"] == "checked" and agree["confidence"] > 0.6


# ------------------------------------------------------------------ text intent
@pytest.mark.parametrize("text,expected", [
    ("Blue floral kurti", {"category": "kurtas-kurtis", "pattern": "floral", "colour": "blue"}),
    ("Maroon embroidered lehenga", {"category": "lehengas", "pattern": "embroidered", "colour": "maroon", "embroidery": "yes"}),
    ("White long dress with full sleeves", {"category": "western-dresses", "sleeve_type": "full sleeves", "length": "ankle length", "colour": "white"}),
    ("Green cotton-looking kurta with a round neck", {"category": "kurtas-kurtis", "neckline": "round neck", "fabric_appearance": "cotton", "colour": "green"}),
    ("Similar design in a darker blue colour", {"colour": "blue", "shade": "dark"}),
    ("Find a similar design in blue colour with full sleeves.", {"colour": "blue", "sleeve_type": "full sleeves"}),
    ("flroal kurthi in marron", {"category": "kurtas-kurtis", "pattern": "floral", "colour": "maroon"}),
    ("saree without embroidery", {"category": "sarees", "embroidery": "no"}),
    ("something nice for my sister's wedding", {}),
])
def test_text_parser(text, expected):
    assert parse_text_query(text)["attributes"] == expected


def test_unknown_words_never_become_filters():
    r = parse_text_query("gorgeous elegant trendy outfit with sleeves")
    assert "sleeve_type" not in r["attributes"] and r["attributes"] == {}


# ------------------------------------------------------------------ model management
def test_model_failure_is_reported_not_crashing():
    from app.models.encoders import EncoderRegistry, ModelUnavailableError

    reg = EncoderRegistry(1)

    def broken():
        raise OSError("weights missing")
    reg.register_override("dinov2-b14", broken)
    with pytest.raises(ModelUnavailableError, match="could not be loaded"):
        reg.get("dinov2-b14")
    with pytest.raises(ModelUnavailableError):  # cooldown: no repeated slow attempts
        reg.get("dinov2-b14")


def test_model_dimension_mismatch_is_refused():
    from app.models.encoders import CVBaselineEncoder, EncoderRegistry, ModelUnavailableError

    class Wrong(CVBaselineEncoder):
        def embed_images(self, images):
            return np.ones((len(images), 128), np.float32)
    reg = EncoderRegistry(1)
    reg.register_override("cv-baseline", Wrong)
    with pytest.raises(ModelUnavailableError, match="Refusing to mix"):
        reg.get("cv-baseline")


def test_unpinned_model_versions_stay_backward_compatible():
    from app.models.encoders import MODEL_SPECS

    assert MODEL_SPECS["fashion-clip"]["version"] == "hf:patrickjohncyh/fashion-clip"  # existing embeddings stay valid


# ------------------------------------------------------------------ retrieval (database)
@requires_db
def test_knn_rejects_wrong_dimensions_and_nan():
    from app.services.db import connection
    from app.services.vector_store import knn

    with connection() as c:
        with pytest.raises(ValueError):
            knn(c, "cv-baseline", "cv-v2", np.ones(128, np.float32))
        bad = np.ones(512, np.float32); bad[0] = np.nan
        with pytest.raises(ValueError):
            knn(c, "cv-baseline", "cv-v2", bad)


@requires_db
def test_settings_do_not_leak_and_cached_plans_are_not_reused():
    from app.models.encoders import MODEL_SPECS
    from app.services.db import connection
    from app.services.vector_store import knn

    ver = MODEL_SPECS["cv-baseline"]["version"]
    with connection() as c:
        row = c.execute("SELECT embedding FROM image_embeddings WHERE model_key='cv-baseline' AND model_version=%s LIMIT 1",
                        (ver,)).fetchone()
        v = row["embedding"].to_numpy()
        for _ in range(8):  # more than psycopg's auto-prepare threshold (5)
            knn(c, "cv-baseline", ver, v, k=50, exact=True)
        assert c.execute("SHOW enable_indexscan").fetchone()["enable_indexscan"] == "on"
        cat = c.execute("""SELECT c.slug, count(*) AS n FROM products p JOIN categories c ON c.id=p.category_id
                           WHERE p.is_active GROUP BY 1 ORDER BY 2 LIMIT 1""").fetchone()
        info: dict = {}
        rows = knn(c, "cv-baseline", ver, v, k=50, filters={"category": [cat["slug"]]}, info=info)
        assert len(rows) == min(50, cat["n"])  # every matching product, not a truncated approximate set
        assert all(isinstance(r["cosine"], float) for r in rows)


@requires_db
def test_filtered_search_returns_only_matching_active_products():
    from app.models.encoders import MODEL_SPECS
    from app.services.db import connection
    from app.services.vector_store import knn, product_details

    ver = MODEL_SPECS["cv-baseline"]["version"]
    with connection() as c:
        v = c.execute("SELECT embedding FROM image_embeddings WHERE model_key='cv-baseline' AND model_version=%s LIMIT 1",
                      (ver,)).fetchone()["embedding"].to_numpy()
        pid = c.execute("SELECT id FROM products WHERE is_active ORDER BY id LIMIT 1").fetchone()["id"]
        c.execute("UPDATE products SET is_active = FALSE WHERE id = %s", (pid,))
        try:
            ids = [r["product_id"] for r in knn(c, "cv-baseline", ver, v, k=200)]
            assert pid not in ids  # inactive products never appear
            rows = knn(c, "cv-baseline", ver, v, k=50, filters={"colour": ["red"]})
            det = product_details(c, [r["product_id"] for r in rows])
            assert all(det[r["product_id"]]["attributes"].get("colour") == "red" for r in rows)
            assert knn(c, "cv-baseline", ver, v, k=50, filters={"min_price": 10_000_000}) == []
            ex = [r["product_id"] for r in knn(c, "cv-baseline", ver, v, k=20, exclude_product_id=ids[0])]
            assert ids[0] not in ex
        finally:
            c.rollback()


@requires_db
def test_candidate_count_is_respected_end_to_end():
    import io

    from app.config import PROJECT_ROOT
    from app.services.db import connection
    from app.services.preprocess import load_and_validate
    from app.services.search import search_image

    img = next((PROJECT_ROOT / "dataset" / "products").rglob("*.jpg"))
    with connection() as c:
        res = search_image(c, load_and_validate(img.read_bytes()), top_k=10, candidates=120)
    assert res["candidates_considered"] == min(120, res["candidates_considered"]) and res["candidates_considered"] > 50
    assert res["retrieval"]["ef_search"] >= 120
    assert {"embed_ms", "vector_search_ms", "details_ms", "rerank_ms", "attributes_ms"} <= set(res["timings_ms"])
