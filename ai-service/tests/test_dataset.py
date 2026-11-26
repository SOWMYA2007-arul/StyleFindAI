"""Dataset tests: count, files exist, duplicate IDs, broken images, metadata, unwanted categories."""
import csv
import os

import pytest

from conftest import CATALOGUE

pytestmark = pytest.mark.skipif(not CATALOGUE.exists(), reason="run prepare_dataset.py first")
MIN_COUNT = int(os.environ.get("STYLEFIND_MIN_CATALOGUE", "1000"))


def rows():
    with open(CATALOGUE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_catalogue_validates():
    from validate_images import validate

    res = validate(min_count=0)
    assert res["passed"], res["problems"][:10]
    assert res["ok_images"] == res["rows"]


def test_catalogue_count():
    n = len(rows())
    is_fixture = all(r["source_dataset"] == "SYNTHETIC_TEST_FIXTURE" for r in rows())
    if is_fixture:
        pytest.skip(f"synthetic fixture with {n} rows - the >= {MIN_COUNT} rule applies to the real catalogue")
    assert n >= MIN_COUNT


def test_no_duplicate_ids_and_required_columns():
    rs = rows()
    ids = [r["product_id"] for r in rs]
    assert len(ids) == len(set(ids))
    for col in ["product_id", "product_name", "category", "subcategory", "colour", "pattern", "sleeve_type",
                "neckline", "image_path", "source_dataset", "licence_information"]:
        assert col in rs[0]
        assert all((r[col] or "").strip() for r in rs), f"empty {col}"


def test_no_unwanted_categories():
    from _common import EXCLUDED_ARTICLE

    bad_words = ("shoe", "heel", "sandal", "bag", "wallet", "watch", "jewel", "bra", "brief", "night")
    for r in rows():
        assert r["subcategory"] not in EXCLUDED_ARTICLE
        assert not any(w in r["subcategory"].lower() for w in bad_words), r


def test_no_duplicate_images():
    from _common import PRODUCTS_DIR
    from remove_duplicates import find_duplicates

    rs = rows()
    assert find_duplicates([(r["product_id"], PRODUCTS_DIR / r["image_path"]) for r in rs]) == {}
