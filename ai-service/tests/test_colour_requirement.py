"""A colour (and garment type) typed by the shopper is a requirement, not a soft preference."""
from __future__ import annotations

import numpy as np

from app.services.colour import colour_family, colour_family_match
from app.services.ranking import QueryContext, colour_requirement_met, finalize, product_colour


def ctx(requested=None, predicted=None):
    return QueryContext(model_key="cv-baseline", palette=[], pattern_vector=None,
                        predicted=predicted or {}, requested=requested or {})


def prod(pid, colour=None, measured=None, palette_lab=None, category="western-dresses"):
    p = {"id": pid, "category": category, "attributes": {"colour": colour} if colour else {}}
    if measured:
        p["measured_colour"] = measured
    if palette_lab is not None:
        p["colour_palette"] = [{"lab": palette_lab, "weight": 1.0}]
    return p


def scored(p, visual):
    return {"product": p, "cosine": visual, "components": {"visual": visual, "colour": 0.0, "pattern": np.nan,
                                                           "attribute": np.nan}, "matched_attributes": []}


def test_families_follow_shopper_words():
    assert {"green", "bottle green", "mint", "olive", "sea green"} <= colour_family("green")
    assert "mustard" not in colour_family("green") and "beige" not in colour_family("green")
    assert "brown" not in colour_family("red") and "purple" not in colour_family("blue")
    assert "bottle green" in colour_family("dark green")          # alias -> shade -> its whole family
    assert colour_family_match("green", "unknown") != 1.0


def test_evidence_order_catalogue_then_measured_then_photo():
    beige_wall_lab = [80.0, 3.0, 15.0]
    # measured garment colour beats a photo colour distorted by the wall, at any confidence
    assert product_colour(prod(1, measured="bottle green", palette_lab=beige_wall_lab))[0] == "bottle green"
    assert product_colour(prod(2, colour="red", measured="green"))[0] == "red"
    assert product_colour(prod(3, palette_lab=[45.0, -40.0, 25.0]))[1] == "photo colour"


def test_wrong_colour_never_shown_even_if_it_looks_more_similar():
    c = ctx({"colour": "green"})
    mustard = scored(prod(1, measured="mustard"), visual=0.99)      # same shape, wrong colour
    green = scored(prod(2, measured="bottle green"), visual=0.40)
    unknown = scored(prod(3), visual=0.95)                            # colour unknown -> cannot promise it
    out = finalize(c, [mustard, green, unknown], top_k=10)
    assert [r["product"]["id"] for r in out] == [2]
    assert c.colour_requirement == {"colour": "green", "category": None, "checked": 3, "kept": 1, "removed": 2}


def test_typed_garment_type_is_required():
    c = ctx({"colour": "green", "category": "western-dresses"})
    kurta = scored(prod(1, measured="green", category="kurtas-kurtis"), 0.9)
    dress = scored(prod(2, measured="green"), 0.5)
    assert [r["product"]["id"] for r in finalize(c, [kurta, dress], 10)] == [2]


def test_colour_only_request_keeps_photo_garment_type_when_confident():
    sure = {"category": {"value": "western-dresses", "confidence": 0.8}}
    unsure = {"category": {"value": "western-dresses", "confidence": 0.3}}
    kurta, dress = scored(prod(1, measured="green", category="kurtas-kurtis"), 0.9), scored(prod(2, measured="green"), 0.5)
    assert [r["product"]["id"] for r in finalize(ctx({"colour": "green"}, sure), [kurta, dress], 10)] == [2]
    assert {r["product"]["id"] for r in finalize(ctx({"colour": "green"}, unsure), [kurta, dress], 10)} == {1, 2}


def test_no_colour_typed_changes_nothing():
    c = ctx({})
    rows = [scored(prod(1, measured="mustard"), 0.9), scored(prod(2, measured="green"), 0.5)]
    assert [r["product"]["id"] for r in finalize(c, rows, 10)] == [1, 2] and c.colour_requirement is None
    assert colour_requirement_met(c, prod(5, measured="mustard"))


def test_search_api_green_request_returns_only_green(monkeypatch):
    """End to end on the test catalogue: every result's colour evidence is in the green family."""
    import pytest

    from test_api import PRODUCTS, _catalogue, search

    row = next((r for r in _catalogue() if r["colour"].lower() in ("white", "off white", "beige", "cream")), None)
    if row is None:
        pytest.skip("no light-coloured product in the test catalogue")
    res = search(PRODUCTS / row["image_path"], top_k=12, text="the same in green colour")
    assert res["colour_requirement"]["colour"] == "green"
    fam = colour_family("green")
    for r in res["results"]:
        assert (r.get("attributes") or {}).get("colour") in fam or "green" in r["name"].lower() \
            or "olive" in r["name"].lower() or "teal" in r["name"].lower() or "mint" in r["name"].lower(), r["name"]
