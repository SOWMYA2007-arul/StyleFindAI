"""Hybrid re-ranking of the top-50 pgvector candidates.

final = w_v*visual + w_c*colour + w_p*pattern + w_a*attribute   (weights from .env, re-normalised)

* Every component is mapped to [0, 1] first, so the weights mean what they say.
* A component that cannot be computed (e.g. the product has no pattern metadata) is NaN; it is
  dropped and the remaining weights are re-normalised -> missing data is not a penalty.
* The score is a *ranking* score, not "the probability that this is the same product".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.config import get_settings
from app.services.colour import (colour_family_match, colour_name_lab, lab_similarity, nearest_colour_name,
                                 palette_similarity, requested_colour_lab)
from app.services.pattern import pattern_similarity

# Cosine similarity between two *unrelated* images is far above 0 for CLIP-like models, so we
# rescale  [floor, 1] -> [0, 1]. Floors are a first guess; tune them with scripts/evaluate.py.
VISUAL_FLOOR = {"fashion-clip": 0.50, "openclip-b32": 0.50, "dinov2-b14": 0.10, "cv-baseline": 0.40}
# Raw cosine needed (plus a margin over the 2nd result) before we call something an exact match.
EXACT_COSINE = {"fashion-clip": 0.96, "openclip-b32": 0.96, "dinov2-b14": 0.92, "cv-baseline": 0.985}
EXACT_MARGIN = 0.02
PHASH_EXACT_BITS = 6      # query is (nearly) the same photo as the catalogue image
# pHash is colour-blind, so a pHash match must also pass a colour check on 32x32 LAB thumbnails:
# Thresholds measured with scripts on resized / re-compressed / +-10% brightness copies (same image)
# versus the same garment with stripes or dots added and other products (see docs/EVALUATION.md):
THUMB_MAX_CHROMA_DIFF = 3.0    # mean |a|,|b| difference on 32x32 LAB thumbnails (same image: <= 2.1)
THUMB_MAX_LIGHT_DIFF = 10.0    # mean |L| difference (same image incl. -10% brightness: <= 8.6)
MIN_SSIM = 0.95                # structural similarity of brightness-normalised 96x128 greyscale (same: >= 0.96)

CATEGORY_AFFINITY = {
    frozenset({"kurtas-kurtis", "other-ethnic"}): 0.5, frozenset({"lehengas", "sarees"}): 0.5,
    frozenset({"western-dresses", "tops-shirts"}): 0.4, frozenset({"kurtas-kurtis", "western-dresses"}): 0.3,
    frozenset({"lehengas", "other-ethnic"}): 0.4, frozenset({"kurtas-kurtis", "tops-shirts"}): 0.3,
}
CATEGORY_KEEP_CONFIDENCE = 0.55  # "this in green" keeps the photo's garment type when the AI is this sure
PRINT_FAMILY = {"printed", "floral", "paisley", "geometric", "abstract", "ethnic motifs", "polka dot",
                "bandhani", "ikat", "leheriya"}


@dataclass
class QueryContext:
    model_key: str
    palette: list[dict]
    pattern_vector: np.ndarray | None
    predicted: dict[str, dict]                       # attr -> {"value", "confidence", "source"}
    requested: dict[str, str] = field(default_factory=dict)  # parsed from the user's text
    colour_requirement: dict | None = None  # set by finalize() when the text asks for a colour
    phash: object | None = None                      # imagehash.ImageHash of the query
    thumb_lab: np.ndarray | None = None              # 32x32 LAB thumbnail of the query
    gray: np.ndarray | None = None                   # normalised 96x128 greyscale of the query (SSIM check)


def lab_thumbnail(img) -> np.ndarray:
    from PIL import Image

    from app.services.foreground import rgb_to_lab

    small = np.asarray(img.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.uint8)
    return rgb_to_lab(small)


def ssim_gray(img) -> np.ndarray:
    from PIL import Image

    a = np.asarray(img.convert("L").resize((96, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def _identity_check(ctx: "QueryContext", storage_key: str) -> bool:
    """True only if the query is the same photo as this catalogue image: colours match (pHash is
    colour-blind) AND fine structure matches (pHash ignores small details such as added stripes)."""
    if ctx.thumb_lab is None:
        return False
    from PIL import Image
    from skimage.metrics import structural_similarity

    try:
        with Image.open(get_settings().products_path / storage_key) as im:
            im = im.convert("RGB")
            cat_lab, cat_gray = lab_thumbnail(im), ssim_gray(im)
    except OSError:
        return False
    d = np.abs(ctx.thumb_lab - cat_lab)
    if float(d[..., 1:].mean()) > THUMB_MAX_CHROMA_DIFF or float(d[..., 0].mean()) > THUMB_MAX_LIGHT_DIFF:
        return False
    if ctx.gray is None:
        return False
    rng = float(cat_gray.max() - cat_gray.min()) or 1.0
    return float(structural_similarity(ctx.gray, cat_gray, data_range=rng)) >= MIN_SSIM


def normalise_weights(w: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, v) for v in w.values()) or 1.0
    return {k: max(0.0, v) / total for k, v in w.items()}


def weights() -> dict[str, float]:
    s = get_settings()
    return normalise_weights({"visual": s.rank_w_visual, "colour": s.rank_w_colour,
                              "pattern": s.rank_w_pattern, "attribute": s.rank_w_attribute})


def normalise_visual(cosine: float, model_key: str) -> float:
    floor = VISUAL_FLOOR.get(model_key, 0.0)
    return float(np.clip((cosine - floor) / (1.0 - floor), 0.0, 1.0))


def category_similarity(a: str | None, b: str | None) -> float:
    if not a or not b or a == "unknown" or b == "unknown":
        return math.nan
    return 1.0 if a == b else CATEGORY_AFFINITY.get(frozenset({a, b}), 0.0)


def value_similarity(attr: str, q: str | None, p: str | None) -> float:
    if not q or not p or q == "unknown" or p == "unknown":
        return math.nan
    if q == p:
        return 1.0
    if attr == "pattern" and q in PRINT_FAMILY and p in PRINT_FAMILY:
        return 0.5
    if attr == "sleeve_type" and {q, p} <= {"full sleeves", "three-quarter"}:
        return 0.5
    return 0.0


def attribute_score(ctx: QueryContext, product: dict) -> tuple[float, list[str]]:
    """Weighted agreement over attributes known on BOTH sides. Requested (text) attributes count more."""
    pattrs = product.get("attributes", {})
    num = den = 0.0
    matched: list[str] = []
    cat_q = ctx.requested.get("category") or ctx.predicted.get("category", {}).get("value")
    s = category_similarity(cat_q, product.get("category"))
    if not math.isnan(s):
        wc = 4.0 if "category" in ctx.requested else 2.0
        num += wc * s; den += wc
    for attr in ("pattern", "sleeve_type", "neckline"):
        q = ctx.requested.get(attr) or ctx.predicted.get(attr, {}).get("value")
        w = 3.0 if attr in ctx.requested else 1.0
        sim = value_similarity(attr, q, pattrs.get(attr))
        if not math.isnan(sim):
            num += w * sim; den += w
            if sim == 1.0:
                matched.append(f"{attr.replace('_', ' ')}: {pattrs.get(attr)}")
    return (num / den if den else math.nan), matched


def product_colour(product: dict) -> tuple[str | None, str]:
    """Best available evidence for a product's colour NAME, most trustworthy first:
    verified catalogue colour > measured garment colour (importers) > whole-photo pixel measurement."""
    for name, label in ((product.get("attributes", {}).get("colour"), "catalogue colour"),
                        (product.get("measured_colour"), "measured garment colour")):
        if name and name != "unknown" and colour_name_lab(name) is not None:
            return name, label
    pal = product.get("colour_palette") or []
    if pal:
        return nearest_colour_name(np.asarray(pal[0]["lab"], dtype=float)), "photo colour"
    pc = product.get("pixel_colour")
    return (pc if pc and colour_name_lab(pc) is not None else None), "photo colour"


def colour_requirement_met(ctx: QueryContext, product: dict) -> bool:
    """A colour typed by the shopper is a requirement, not a preference."""
    req = ctx.requested.get("colour")
    if not req:
        return True
    name, _ = product_colour(product)
    return colour_family_match(req, name) == 1.0   # unknown colour -> cannot promise it -> excluded


def colour_score(ctx: QueryContext, product: dict) -> float:
    req = ctx.requested.get("colour")
    if req:
        target = requested_colour_lab(req, ctx.requested.get("shade"))
        name, _ = product_colour(product)
        return lab_similarity(target, colour_name_lab(name) if name else None, sigma=25.0)  # shades within the family
    return palette_similarity(ctx.palette, product.get("colour_palette") or [])


def combine(components: dict[str, float], w: dict[str, float] | None = None) -> float:
    """Weighted mean of the AVAILABLE components (NaN / None are skipped, weights renormalised)."""
    w = normalise_weights(w) if w else weights()
    avail = {k: v for k, v in components.items() if v is not None and not math.isnan(v) and k in w}
    tw = sum(w[k] for k in avail) or 1.0
    return float(sum(w[k] * v for k, v in avail.items()) / tw)


def score_candidates(ctx: QueryContext, candidates: list[dict], details: dict[int, dict]) -> list[dict]:
    """Component scores for every candidate (independent of the weights, so they can be cached
    and re-weighted offline by scripts/tune_weights.py)."""
    out = []
    for c in candidates:
        p = details.get(c["product_id"])
        if p is None:
            continue
        cos = float(c["cosine"])
        attr, matched = attribute_score(ctx, p)
        comps = {"visual": normalise_visual(cos, ctx.model_key), "colour": colour_score(ctx, p),
                 "pattern": pattern_similarity(ctx.pattern_vector, p.get("pattern_vector")), "attribute": attr}
        out.append({"product": p, "cosine": cos, "components": comps, "matched_attributes": matched})
    return out


def sort_key(r: dict):
    # final score, then raw visual similarity, then product id -> fully deterministic order
    return (-round(r["final"], 9), -round(r["cosine"], 9), r["product"]["id"])


def finalize(ctx: QueryContext, scored: list[dict], top_k: int, w: dict[str, float] | None = None) -> list[dict]:
    required_cat = ctx.requested.get("category")  # "... dress" / "... kurta": the typed garment type is required
    if not required_cat and ctx.requested.get("colour"):
        # "this in green": only the colour changes, so the garment type of the photo stays - if the AI is fairly
        # sure what it is (otherwise no category requirement at all)
        pc = ctx.predicted.get("category") or {}
        if pc.get("value") not in (None, "unknown") and float(pc.get("confidence") or 0) >= CATEGORY_KEEP_CONFIDENCE:
            required_cat = pc["value"]
    if required_cat:
        scored = [r for r in scored if r["product"].get("category") == required_cat]
    if ctx.requested.get("colour"):
        kept = [r for r in scored if colour_requirement_met(ctx, r["product"])]
        ctx.colour_requirement = {"colour": " ".join(x for x in (ctx.requested.get("shade"), ctx.requested["colour"]) if x),
                                  "category": required_cat, "checked": len(scored), "kept": len(kept),
                                  "removed": len(scored) - len(kept)}
        scored = kept
    results = [dict(r, final=combine(r["components"], w)) for r in scored]
    results.sort(key=sort_key)
    _assign_match_types(ctx, results)
    for r in results:
        r["reasons"] = explain(ctx, r)
    return results[:top_k]


def rerank(ctx: QueryContext, candidates: list[dict], details: dict[int, dict], top_k: int,
           w: dict[str, float] | None = None) -> list[dict]:
    return finalize(ctx, score_candidates(ctx, candidates, details), top_k, w)


def _assign_match_types(ctx: QueryContext, results: list[dict]) -> None:
    by_cos = sorted(results, key=lambda r: (-r["cosine"], r["product"]["id"]))
    second = by_cos[1]["cosine"] if len(by_cos) > 1 else 0.0
    for r in results:
        r["match_type"] = "similar" if r["components"]["visual"] >= 0.5 else "related"
        r["exact_evidence"] = None
    if not by_cos:
        return
    top = by_cos[0]
    evidence = None
    if ctx.phash is not None:
        import imagehash

        # Identity check on the 5 visually closest items. If two catalogue items both look
        # identical to the query we cannot tell which one it is -> no exact claim at all.
        passing = []
        for r in by_cos[:5]:
            if not r["product"].get("phash"):
                continue
            bits = ctx.phash - imagehash.hex_to_hash(r["product"]["phash"])
            if bits <= PHASH_EXACT_BITS and _identity_check(ctx, r["product"]["storage_key"]):
                passing.append((r, bits))
        if len(passing) == 1:
            top, bits = passing[0]
            evidence = f"near-identical photo (hash differs by {bits}/64 bits; colours and fine detail match)"
        elif len(passing) > 1:
            return
    if evidence is None and top["cosine"] >= EXACT_COSINE.get(ctx.model_key, 1.01) \
            and top["cosine"] - second >= EXACT_MARGIN:
        evidence = f"very high visual similarity ({top['cosine']:.3f}) clearly above the next product"
    if evidence and not ctx.requested:  # with a text modification the user wants something different
        top["match_type"] = "exact"
        top["exact_evidence"] = evidence


def explain(ctx: QueryContext, r: dict) -> list[str]:
    c = r["components"]; p = r["product"]
    out = []
    if r["match_type"] == "exact":
        out.append(f"Likely the same product: {r['exact_evidence']}")
    v = c["visual"]
    out.append(("Very similar overall look" if v >= 0.8 else "Similar overall look" if v >= 0.5
                else "Some visual resemblance") + f" (visual score {v:.2f})")
    col = c["colour"]
    if not math.isnan(col) and col >= 0.6:
        name = (p.get("attributes", {}).get("colour")
                or p.get("measured_colour")
                or p.get("pixel_colour"))
        if ctx.requested.get("colour"):
            want = " ".join(x for x in (ctx.requested.get("shade"), ctx.requested["colour"]) if x)
            out.append(f"Matches requested colour '{want}' ({name})")
        else:
            out.append(f"Similar colours ({name})")
    pat = c["pattern"]
    if not math.isnan(pat) and pat >= 0.75:
        out.append(f"Similar pattern/texture ({p.get('attributes', {}).get('pattern') or p.get('pattern_label')})")
    qcat = ctx.predicted.get("category", {}).get("value")
    if qcat and qcat == p.get("category"):
        out.append(f"Same category: {p.get('category_name')}")
    for m in r["matched_attributes"]:
        out.append(f"Same {m}")
    return out
