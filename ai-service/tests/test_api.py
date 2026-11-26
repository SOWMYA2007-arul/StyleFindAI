"""Integration tests through the real HTTP API, real PostgreSQL + pgvector, real stored embeddings.

These include the 8 scenarios from the brief. Queries come from scripts/make_eval_queries.py
(transformed catalogue photos with a known answer).
"""
import csv
import json

import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, get_settings
from app.main import app
from conftest import QUERIES, image_bytes, requires_db

pytestmark = requires_db
client = TestClient(app)
PRODUCTS = PROJECT_ROOT / "dataset" / "products"


def _catalogue():
    with open(PROJECT_ROOT / "dataset" / "metadata" / "catalogue.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def search(path, **form):
    with open(path, "rb") as f:
        r = client.post("/ai/search", files={"image": (path.name, f, "image/jpeg")},
                        data={k: (json.dumps(v) if isinstance(v, dict) else str(v)) for k, v in form.items()})
    assert r.status_code == 200, r.text
    return r.json()


def test_health():
    r = client.get("/health").json()
    assert r["database"] == "ok" and r["pgvector"] and r["embeddings"][get_settings().ai_primary_model] > 0


def test_catalogue_photo_finds_itself_as_exact_match():
    row = _catalogue()[0]
    res = search(PRODUCTS / row["image_path"], top_k=10)
    top = res["results"][0]
    assert top["product_code"] == row["product_id"]
    assert top["match_type"] == "exact" and any("same product" in x for x in top["reasons"])
    assert 10 >= len(res["results"]) and res["candidates_considered"] >= 10
    assert all(not r["image_key"].startswith(("/", "C:")) for r in res["results"])  # no absolute paths


def test_results_are_real_db_records_sorted_by_score():
    row = _catalogue()[5]
    res = search(PRODUCTS / row["image_path"], top_k=20)
    finals = [r["scores"]["final"] for r in res["results"]]
    assert finals == sorted(finals, reverse=True)
    with __import__("app.services.db", fromlist=["connection"]).connection() as c:  # every result is a real DB row
        codes = {r["product_code"] for r in c.execute("SELECT product_code FROM products WHERE is_active")}
    assert all(r["product_code"] in codes for r in res["results"])
    assert all((PRODUCTS / r["image_key"]).exists() for r in res["results"])


def test_invalid_uploads_rejected():
    r = client.post("/ai/search", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422 and "valid image" in r.json()["error"]
    r = client.post("/ai/search", files={"image": ("x.gif", image_bytes("GIF"), "image/gif")})
    assert r.status_code == 422
    r = client.post("/ai/search", files={"image": ("x.jpg", image_bytes(), "image/jpeg")}, data={"filters": "{bad"})
    assert r.status_code == 422
    r = client.post("/ai/search", files={"image": ("x.jpg", image_bytes(), "image/jpeg")}, data={"model": "nope"})
    assert r.status_code == 422


def test_category_filter_is_applied_in_database():
    row = _catalogue()[0]
    res = search(PRODUCTS / row["image_path"], top_k=20, filters={"category": ["sarees"]})
    assert res["results"] and all(r["category"] == "sarees" for r in res["results"])


def test_text_modification_changes_colour_ranking():
    row = next(r for r in _catalogue() if r["colour"].lower() not in ("blue", "navy blue"))
    # a wide candidate pool, so the test measures the text edit, not how many products the catalogue holds
    plain = search(PRODUCTS / row["image_path"], top_k=10, candidates=300)
    edited = search(PRODUCTS / row["image_path"], top_k=10, candidates=300, text="similar design in blue colour")
    assert edited["query"]["requested_attributes"]["colour"] == "blue"
    assert edited["query"]["composition"]["method"] in ("text_delta", "attribute_rerank_only")
    blue = lambda res: sum(r["colour"] in ("blue", "navy blue", "turquoise blue") for r in res["results"])  # noqa: E731
    assert blue(edited) > blue(plain)
    assert all(r["match_type"] != "exact" for r in edited["results"])


def test_similar_products_endpoint():
    with __import__("app.services.db", fromlist=["connection"]).connection() as c:
        pid = c.execute("SELECT id FROM products ORDER BY id LIMIT 1").fetchone()["id"]
    r = client.get(f"/ai/products/{pid}/similar?top_k=8").json()
    assert len(r["results"]) == 8 and all(x["product_id"] != pid for x in r["results"])
    assert client.get("/ai/products/999999999/similar").status_code == 404


def test_extract_attributes_and_embed():
    row = _catalogue()[0]
    with open(PRODUCTS / row["image_path"], "rb") as f:
        a = client.post("/ai/extract-attributes", files={"image": ("q.jpg", f, "image/jpeg")}).json()
    assert a["attributes"]["colour"]["value"] != "" and a["palette"]
    with open(PRODUCTS / row["image_path"], "rb") as f:
        e = client.post("/ai/embed", files={"image": ("q.jpg", f, "image/jpeg")}).json()
    assert e["dims"] == len(e["embedding"])


def test_index_endpoint_requires_internal_token():
    assert client.post("/ai/index", json={"product_ids": [1]}).status_code == 401
    r = client.post("/ai/index", json={"product_ids": [1]}, headers={"x-internal-token": get_settings().ai_internal_token})
    assert r.status_code == 200 and r.json()["indexed"] == 0  # already embedded -> nothing to do


def test_compare_models():
    row = _catalogue()[0]
    with open(PRODUCTS / row["image_path"], "rb") as f:
        r = client.post("/ai/compare-models", files={"image": ("q.jpg", f, "image/jpeg")}).json()
    assert get_settings().ai_primary_model in r["models_compared"]


# ------------------------------------------------ the 8 AI scenarios from the brief
def _queries():
    if not QUERIES.exists():
        pytest.skip("run scripts/make_eval_queries.py first")
    with open(QUERIES, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _recall_at(transform, k=10, limit=25):
    qs = [q for q in _queries() if q["transform"] == transform][:limit]
    hits = 0
    for q in qs:
        res = search(PROJECT_ROOT / "dataset" / "evaluation" / q["query_path"], top_k=k)
        hits += q["product_id"] in [r["product_code"] for r in res["results"]]
    return hits / len(qs)


# Thresholds are deliberately modest so they hold for the weak cv-baseline too; FashionCLIP
# should beat them comfortably on the real catalogue (see docs/EVALUATION.md for measured values).
@pytest.mark.parametrize("transform,scenario,min_recall", [
    ("background", "1. same product, different background", 0.6),
    ("angle", "2. same product, different angle", 0.6),
    ("lighting", "6. customer-style photo vs catalogue photo", 0.6),
    ("partial", "7. partial garment photograph", 0.5),
    ("clutter", "8. photograph containing multiple objects", 0.6),
])
def test_scenario_recall_at_10(transform, scenario, min_recall):
    assert _recall_at(transform) >= min_recall, scenario


def _consistency(attr, other_attr):
    """Scenario 3/4: for queries, measure how often top-10 share `attr` with the source product,
    and how often they DIFFER in `other_attr` (i.e. the system also returns variety)."""
    cat = {r["product_id"]: r for r in _catalogue()}
    same, diff, n = 0, 0, 0
    for q in [q for q in _queries() if q["transform"] == "angle"][:20]:
        src = cat[q["product_id"]]
        res = search(PROJECT_ROOT / "dataset" / "evaluation" / q["query_path"], top_k=10)
        for r in res["results"][1:]:
            n += 1
            same += r[attr] == src[attr].lower()
            diff += r[other_attr] != src[other_attr].lower()
    return same / n, diff / n


def test_scenario_3_similar_colours_different_patterns():
    same_colour, different_pattern = _consistency("colour", "pattern")
    base = _base_rate("colour")
    assert same_colour > 2 * base, (same_colour, base)   # far above chance
    assert different_pattern > 0.2                       # still shows other patterns


def test_scenario_4_same_pattern_different_colours():
    same_pattern, different_colour = _consistency("pattern", "colour")
    assert same_pattern > _base_rate("pattern"), same_pattern
    assert different_colour > 0.1


def test_scenario_5_similar_category():
    cat = {r["product_id"]: r for r in _catalogue()}
    hits = n = 0
    for q in [q for q in _queries() if q["transform"] == "background"][:20]:
        res = search(PROJECT_ROOT / "dataset" / "evaluation" / q["query_path"], top_k=10)
        n += len(res["results"])
        hits += sum(r["category"] == cat[q["product_id"]]["category"] for r in res["results"])
    assert hits / n > _base_rate("category")


def _base_rate(attr):
    from collections import Counter

    c = Counter((r[attr] or "").lower() for r in _catalogue())
    total = sum(c.values())
    return sum((v / total) ** 2 for v in c.values())  # chance that two random products share the value
