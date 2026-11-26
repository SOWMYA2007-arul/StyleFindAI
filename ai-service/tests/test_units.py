"""Fast unit tests - no database, no model downloads."""
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services.colour import extract_palette, name_similarity, normalise_colour_name, palette_similarity
from app.services.pattern import pattern_descriptor, pattern_similarity
from app.services.preprocess import ImageValidationError, load_and_validate
from app.services.ranking import QueryContext, attribute_score, combine, normalise_visual, value_similarity
from app.utils.keywords import attributes_from_name, attributes_from_text_query
from conftest import image_bytes


# ---------------- preprocessing / upload validation
@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_accepts_supported_formats(fmt):
    li = load_and_validate(image_bytes(fmt))
    assert li.image.mode == "RGB" and li.width == 200 and len(li.sha256) == 64


def test_transparent_png_becomes_white_rgb():
    li = load_and_validate(image_bytes("PNG", mode="RGBA", colour=(0, 0, 0)))
    assert li.image.mode == "RGB" and li.image.getpixel((5, 5)) == (255, 255, 255)


@pytest.mark.parametrize("data,msg", [
    (b"", "empty"), (b"hello I am a text file", "not a valid image"),
])
def test_rejects_non_images(data, msg):
    with pytest.raises(ImageValidationError, match=msg):
        load_and_validate(data)


def test_rejects_gif_tiny_and_oversized():
    with pytest.raises(ImageValidationError, match="Only JPG"):
        load_and_validate(image_bytes("GIF"))
    with pytest.raises(ImageValidationError, match="too small"):
        load_and_validate(image_bytes(size=(30, 30)))
    with pytest.raises(ImageValidationError, match="larger than"):
        load_and_validate(image_bytes(size=(400, 400)), max_bytes=100)


def test_quality_warning_for_dark_image():
    li = load_and_validate(image_bytes(colour=(5, 5, 5)))
    assert any("dark" in w for w in li.quality_warnings)


# ---------------- colour
def _garment(colour, bg=(248, 248, 248), stripes=None):
    im = Image.new("RGB", (200, 260), bg)
    d = ImageDraw.Draw(im)
    d.polygon([(60, 20), (140, 20), (170, 240), (30, 240)], fill=colour)
    if stripes:
        for y in range(20, 240, 16):
            d.rectangle((30, y, 170, y + 5), fill=stripes)
        d.polygon([(0, 0), (59, 0), (29, 259), (0, 259)], fill=bg)
        d.polygon([(141, 0), (199, 0), (199, 259), (171, 259)], fill=bg)
    return im


def test_palette_finds_garment_colour_not_background():
    pal = extract_palette(_garment((200, 30, 40)))
    assert pal[0]["name"] == "red" and pal[0]["weight"] > 0.6


def test_palette_similarity_orders_colours():
    red, red2, blue = (extract_palette(_garment(c)) for c in [(200, 30, 40), (190, 40, 45), (40, 90, 200)])
    assert palette_similarity(red, red2) > 0.8 > palette_similarity(red, blue)
    assert math.isnan(palette_similarity([], red))


def test_colour_names():
    assert normalise_colour_name("Navy") == "navy blue" and normalise_colour_name("Grey Melange") == "grey"
    assert normalise_colour_name("plaid-ish") is None
    assert name_similarity("navy blue", "blue") > name_similarity("navy blue", "yellow")


# ---------------- pattern
def test_pattern_descriptor_separates_solid_and_striped():
    solid, lab_s = pattern_descriptor(_garment((40, 90, 200)))
    solid2, _ = pattern_descriptor(_garment((200, 30, 40)))
    striped, lab_p = pattern_descriptor(_garment((40, 90, 200), stripes=(250, 250, 250)))
    assert lab_s == "solid" and lab_p == "patterned"
    assert pattern_similarity(solid, solid2) > pattern_similarity(solid, striped)
    assert math.isnan(pattern_similarity(None, solid))


# ---------------- keywords
def test_attributes_from_product_name():
    a = attributes_from_name("Women Blue Printed Straight Kurta with 3/4 Sleeves and Mandarin Collar")
    assert a == {"pattern": "printed", "sleeve_type": "three-quarter", "neckline": "mandarin collar",
                 "embroidery": "unknown"}
    assert attributes_from_name("Women Pink Kurta")["pattern"] == "unknown"


def test_text_query_parsing():
    r = attributes_from_text_query("Find a similar design in blue colour with full sleeves.")
    assert r["colour"] == "blue" and r["sleeve_type"] == "full sleeves"
    assert attributes_from_text_query("same but in navy blue")["colour"] == "navy blue"


# ---------------- ranking
def test_combine_ignores_missing_components():
    full = combine({"visual": 0.8, "colour": 0.8, "pattern": 0.8, "attribute": 0.8})
    missing = combine({"visual": 0.8, "colour": 0.8, "pattern": float("nan"), "attribute": float("nan")})
    assert abs(full - 0.8) < 1e-9 and abs(missing - 0.8) < 1e-9  # no penalty for unknown data


def test_visual_normalisation_is_bounded():
    assert normalise_visual(1.0, "fashion-clip") == 1.0
    assert normalise_visual(0.2, "fashion-clip") == 0.0
    assert 0 < normalise_visual(0.75, "fashion-clip") < 1


def test_attribute_score_unknown_not_penalised():
    ctx = QueryContext("cv-baseline", [], None, {"category": {"value": "sarees"}, "pattern": {"value": "unknown"}})
    s, _ = attribute_score(ctx, {"category": "sarees", "attributes": {"pattern": "floral"}})
    assert s == 1.0
    s2, _ = attribute_score(ctx, {"category": "tops-shirts", "attributes": {}})
    assert s2 == 0.0
    assert value_similarity("pattern", "floral", "printed") == 0.5


def test_requested_attributes_dominate():
    ctx = QueryContext("cv-baseline", [], None, {"sleeve_type": {"value": "short sleeves"}},
                       requested={"sleeve_type": "full sleeves"})
    good, _ = attribute_score(ctx, {"category": "x", "attributes": {"sleeve_type": "full sleeves"}})
    bad, _ = attribute_score(ctx, {"category": "x", "attributes": {"sleeve_type": "short sleeves"}})
    assert good > bad


def test_cv_baseline_encoder_shape():
    from app.models.encoders import CVBaselineEncoder

    v = CVBaselineEncoder().embed_images([_garment((200, 30, 40)), _garment((40, 90, 200))])
    assert v.shape == (2, 512) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
