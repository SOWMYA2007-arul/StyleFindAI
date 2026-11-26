"""Exact-vs-similar classification scenarios from the brief, through the real pipeline + database.

An 'exact' label must only appear when the query is (nearly) the same catalogue photo of exactly one
product. Everything else must be 'similar' / 'related'.
"""
import csv
import io

import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from app.config import PROJECT_ROOT
from app.services.db import connection
from app.services.preprocess import load_and_validate
from app.services.search import search_image
from conftest import requires_db

pytestmark = requires_db
PRODUCTS = PROJECT_ROOT / "dataset" / "products"


def _row(i=0):
    with open(PROJECT_ROOT / "dataset" / "metadata" / "catalogue.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[i]


def _search(img: Image.Image) -> list[dict]:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    with connection() as conn:
        return search_image(conn, load_and_validate(buf.getvalue()), top_k=10)["results"]


def _exact(results):
    return [r["product_code"] for r in results if r["match_type"] == "exact"]


@pytest.fixture(scope="module")
def base():
    r = _row(3)
    return r, Image.open(PRODUCTS / r["image_path"]).convert("RGB")


def test_identical_image_is_exact(base):
    r, img = base
    assert _exact(_search(img)) == [r["product_id"]]


def test_resized_identical_image_is_exact(base):
    r, img = base
    big = img.resize((img.width * 2, img.height * 2), Image.Resampling.BICUBIC)
    assert _exact(_search(big)) == [r["product_id"]]


def test_slightly_altered_image_is_exact(base):
    r, img = base
    altered = ImageEnhance.Brightness(img).enhance(1.04)
    assert _exact(_search(altered)) == [r["product_id"]]


def test_different_colour_is_never_exact(base):
    r, img = base
    # same garment, recoloured: rotate the colours (R,G,B) -> (B,R,G)
    rgb = img.split()
    recoloured = Image.merge("RGB", (rgb[2], rgb[0], rgb[1]))
    res = _search(recoloured)
    assert _exact(res) == []


def test_different_pattern_is_never_exact(base):
    r, img = base
    patterned = img.copy()
    d = ImageDraw.Draw(patterned)
    for y in range(0, patterned.height, 14):
        d.line([(0, y), (patterned.width, y)], fill=(20, 20, 20), width=4)
    assert _exact(_search(patterned)) == []


def test_mirrored_photo_is_not_claimed_as_exact_for_another_product(base):
    r, img = base
    ex = _exact(_search(ImageOps.mirror(img)))
    assert ex in ([], [r["product_id"]])  # may be the same product, never a different one


def test_unseen_garment_is_never_exact():
    # a garment that is not in the catalogue
    img = Image.new("RGB", (300, 400), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.polygon([(120, 40), (180, 40), (260, 380), (40, 380)], fill=(0, 150, 140))
    for x in range(40, 260, 20):
        d.ellipse((x, 200, x + 8, 208), fill=(250, 220, 0))
    assert _exact(_search(img)) == []


def test_same_category_different_product_not_exact_for_original(base):
    r, img = base
    other = next(x for x in csv.DictReader(open(PROJECT_ROOT / "dataset" / "metadata" / "catalogue.csv", encoding="utf-8"))
                 if x["category"] == r["category"] and x["product_id"] != r["product_id"])
    ex = _exact(_search(Image.open(PRODUCTS / other["image_path"])))
    assert r["product_id"] not in ex


def test_repeated_search_is_stable(base):
    _, img = base
    a = [x["product_code"] for x in _search(img)]
    b = [x["product_code"] for x in _search(img)]
    assert a == b
