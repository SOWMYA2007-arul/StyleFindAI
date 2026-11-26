"""The end-to-end visual search pipeline:

upload -> validate/preprocess -> (optional) isolate garment -> embed with the selected model
-> pgvector top-N (cosine, N = SEARCH_CANDIDATES) -> predict query attributes -> hybrid re-rank -> top 10-20
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import imagehash
import numpy as np

from app.config import get_settings
from app.models.encoders import MODEL_SPECS, get_registry, l2_normalise
from app.services import vector_store
from app.services.attributes import merge_pattern, neighbour_vote, pixel_attributes, vlm_attributes, zero_shot
from app.services.detection import isolate_garment
from app.services.segmentation.cache import LRUCache
from app.services.preprocess import LoadedImage
from app.services.ranking import QueryContext, lab_thumbnail, rerank, ssim_gray
from app.utils.keywords import attributes_from_text_query

COMPOSE_LAMBDA = 1.0  # strength of the text edit in embedding space (tune with evaluation)


def _describe(attrs: dict[str, str]) -> str:
    colour = " ".join(x for x in (attrs.get("shade"), attrs.get("colour")) if x and x != "unknown")
    parts = [colour, attrs.get("pattern", ""),
             (attrs.get("category") or "garment").replace("-", " "),
             f"with {attrs['sleeve_type']}" if attrs.get("sleeve_type") else "",
             f"and {attrs['neckline']}" if attrs.get("neckline") else ""]
    return "a photo of a " + " ".join(p for p in parts if p and p != "unknown")


def compose_query(enc, img_emb: np.ndarray, predicted: dict, requested: dict, text: str) -> tuple[np.ndarray, dict]:
    """Zero-shot composed retrieval with a text *delta*:
        q = norm( image + lambda * (text(target) - text(source)) )
    source = description of what the query shows (from predicted attributes),
    target = the same description with the user's requested changes applied.
    Moving the image embedding along this direction keeps the design but changes the described
    attributes. This is a training-free baseline (not a trained CIR model such as CLIP4Cir)."""
    src = {k: v.get("value") for k, v in predicted.items() if isinstance(v, dict)}
    tgt = dict(src); tgt.update(requested)
    s_txt, t_txt = _describe(src), _describe(tgt) + f", {text}"
    t = enc.embed_texts([s_txt, t_txt])
    q = l2_normalise(img_emb + COMPOSE_LAMBDA * (t[1] - t[0]))
    return q, {"method": "text_delta", "source_text": s_txt, "target_text": t_txt, "lambda": COMPOSE_LAMBDA}


def _format(r: dict) -> dict:
    p, c = r["product"], r["components"]
    clean = lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)  # noqa: E731
    attrs = p.get("attributes", {})
    return {
        "product_id": p["id"], "product_code": p["product_code"], "name": p["name"],
        "category": p["category"], "category_name": p["category_name"], "subcategory": p["subcategory"],
        "colour": attrs.get("colour") or p.get("measured_colour") or "unknown",
        "pattern": attrs.get("pattern", "unknown"),
        "sleeve_type": attrs.get("sleeve_type", "unknown"), "neckline": attrs.get("neckline", "unknown"),
        "price": float(p["price"]) if p.get("price") is not None else None,
        "image_key": p["storage_key"],
        "scores": {"final": clean(r["final"]), "visual": clean(c["visual"]), "visual_cosine": clean(r["cosine"]),
                   "colour": clean(c["colour"]), "pattern": clean(c["pattern"]), "attribute": clean(c["attribute"])},
        "match_type": r["match_type"], "reasons": r["reasons"],
    }

@dataclass
class PreparedQuery:
    """Everything about the uploaded photo that does not depend on the candidate count / weights."""
    loaded: LoadedImage
    enc: object
    image: object                 # PIL image actually encoded (full or isolated garment)
    crop_info: dict
    embedding: np.ndarray         # query vector (after optional text composition / multi-crop)
    image_embedding: np.ndarray   # pure image vector
    pix: dict
    predicted: dict
    requested: dict
    composition: dict | None
    crop_embeddings: list = field(default_factory=list)   # extra multi-crop vectors (may be empty)
    timings: dict = field(default_factory=dict)


_emb_cache: LRUCache | None = None


def embedding_cache() -> LRUCache:
    """Query embeddings keyed by (image sha256, model, model version, crop result, multi-crop mode):
    repeating a search (pagination, filter changes, model comparison) does not re-run the encoder."""
    global _emb_cache
    if _emb_cache is None:
        _emb_cache = LRUCache(get_settings().embedding_cache_size)
    return _emb_cache


def prepare_query(loaded: LoadedImage, model_key: str | None = None, text: str | None = None,
                  crop: bool = False, multicrop: str | None = None, target=None) -> PreparedQuery:
    s = get_settings()
    model_key = model_key or s.ai_primary_model
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    img, crop_info = (isolate_garment(loaded.image, loaded.sha256, target) if crop else
                      (loaded.image, {"method": "full_image", "reason": "crop not requested"}))
    timings["isolate_ms"] = (time.perf_counter() - t0) * 1000

    t = time.perf_counter()
    enc = get_registry().get(model_key)
    timings["model_load_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    mode = (multicrop or s.search_multicrop or "off").lower()
    ekey = (loaded.sha256, enc.key, enc.version, crop_info.get("method"), str(crop_info.get("box")), str(target), mode)
    hit = embedding_cache().get(ekey) if s.embedding_cache_size > 0 else None
    if hit is not None:
        emb, crop_vecs = hit
        timings["embedding_cache_hit"] = 1
    else:
        crops = query_crops(loaded.image, img) if mode != "off" else []
        vecs = enc.embed_images([img] + [c for _, c in crops])
        emb, crop_vecs = vecs[0], list(vecs[1:])
        if s.embedding_cache_size > 0:
            embedding_cache().put(ekey, (emb, crop_vecs))
    timings["embed_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    pix = pixel_attributes(img)
    predicted: dict = {"colour": pix["colour"], "secondary_colour": pix["secondary_colour"],
                       "texture": pix["texture"]}
    if pix.get("pattern"):
        predicted["pattern"] = pix["pattern"]
    if enc.supports_text:
        for k2, v in zero_shot(enc, emb).items():
            if k2 == "pattern" and "pattern" in predicted:
                predicted["pattern"] = merge_pattern(predicted["pattern"], v)
            else:
                predicted[k2] = v
    predicted.update({k2: v for k2, v in vlm_attributes(img).items() if v.get("value") not in (None, "unknown")})
    timings["attributes_ms"] = (time.perf_counter() - t) * 1000

    requested = attributes_from_text_query(text) if text else {}
    composition = None
    query_vec = emb
    if mode == "mean_embedding" and crop_vecs:
        query_vec = l2_normalise(np.mean(np.stack([emb] + crop_vecs), axis=0))
    if text:
        if enc.supports_text:
            t = time.perf_counter()
            query_vec, composition = compose_query(enc, query_vec, predicted, requested, text)
            timings["compose_ms"] = (time.perf_counter() - t) * 1000
        else:
            composition = {"method": "attribute_rerank_only",
                           "note": f"{enc.key} has no text encoder; text is used only to re-rank by attributes"}
    return PreparedQuery(loaded, enc, img, crop_info, query_vec, emb, pix, predicted, requested, composition,
                         crop_vecs if mode == "max_score" else [], timings)


def query_crops(original, garment) -> list[tuple[str, object]]:
    """Extra views for multi-crop retrieval: upper and lower halves of the garment region
    (sleeves/neckline vs hem/border). Kept to 2 extra crops to limit CPU cost."""
    w, h = garment.size
    if min(w, h) < 64:
        return []
    return [("upper", garment.crop((0, 0, w, int(h * 0.6)))), ("lower", garment.crop((0, int(h * 0.4), w, h)))]


def retrieve(conn, pq: PreparedQuery, k: int, filters: dict | None = None, ef_search: int | None = None,
             info: dict | None = None) -> list[dict]:
    """pgvector retrieval; with max_score multi-crop each view retrieves and a product keeps its best cosine."""
    enc = pq.enc
    rows = vector_store.knn(conn, enc.key, enc.version, pq.embedding, k=k, filters=filters, ef_search=ef_search, info=info)
    if not pq.crop_embeddings:
        return rows
    best = {r["product_id"]: r for r in rows}
    for v in pq.crop_embeddings:
        for r in vector_store.knn(conn, enc.key, enc.version, v, k=k, filters=filters, ef_search=ef_search):
            # a crop counts slightly less than the whole garment so it only rescues near-misses
            r = dict(r, cosine=float(r["cosine"]) - 0.02)
            if r["product_id"] not in best or r["cosine"] > float(best[r["product_id"]]["cosine"]):
                best[r["product_id"]] = r
    merged = sorted(best.values(), key=lambda r: (-round(float(r["cosine"]), 6), r["product_id"]))[:k]
    if info is not None:
        info["multicrop_views"] = 1 + len(pq.crop_embeddings)
    return merged


def build_context(pq: PreparedQuery, details: dict, cands: list[dict]) -> QueryContext:
    predicted = dict(pq.predicted)
    if "category" not in predicted or predicted["category"]["value"] == "unknown":
        top = [c for c in cands[:10] if c["product_id"] in details]
        vote = neighbour_vote([details[c["product_id"]]["category"] for c in top], [float(c["cosine"]) for c in top])
        if "category" not in predicted or vote["value"] != "unknown":
            predicted["category"] = vote
    return QueryContext(model_key=pq.enc.key, palette=pq.pix["palette"], pattern_vector=pq.pix["pattern_vector"],
                        predicted=predicted, requested=pq.requested, phash=imagehash.phash(pq.loaded.image),
                        thumb_lab=lab_thumbnail(pq.loaded.image), gray=ssim_gray(pq.loaded.image))


def run_query(conn, pq: PreparedQuery, top_k: int = 20, filters: dict | None = None,
              candidates: int | None = None, ef_search: int | None = None, weights: dict | None = None) -> dict:
    s = get_settings()
    timings = dict(pq.timings)
    k = max(top_k, candidates or s.search_candidates)
    if pq.requested.get("colour"):
        # a typed colour is a requirement: look further than usual so enough items in that colour are found
        k = max(k, s.colour_request_candidates)
    info: dict = {}
    t = time.perf_counter()
    cands = retrieve(conn, pq, k, filters, ef_search, info)
    if pq.requested.get("colour") or pq.requested.get("category"):
        # Requirements from the text: also fetch the visually closest products that CAN satisfy them, so a
        # catalogue whose nearest neighbours are all the wrong colour still returns enough results.
        from app.services.colour import colour_family

        extra = dict(filters or {})
        if pq.requested.get("colour"):
            extra["colour_evidence"] = colour_family(pq.requested["colour"])
        if pq.requested.get("category") and not extra.get("category"):
            extra["category"] = [pq.requested["category"]]
        pool = vector_store.knn(conn, pq.enc.key, pq.enc.version, pq.embedding, k=s.colour_request_candidates // 2,
                                filters=extra, ef_search=ef_search)
        have = {c["product_id"] for c in cands}
        cands = cands + [r for r in pool if r["product_id"] not in have]
        info["requirement_pool"] = len(pool)
    timings["vector_search_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    details = vector_store.product_details(conn, [c["product_id"] for c in cands])
    timings["details_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    ctx = build_context(pq, details, cands)
    ranked = rerank(ctx, cands, details, top_k, weights)
    timings["rerank_ms"] = (time.perf_counter() - t) * 1000
    timings["total_ms"] = sum(v for k2, v in timings.items() if k2.endswith("_ms") and k2 != "total_ms")
    enc, loaded = pq.enc, pq.loaded
    return {
        "model": {"key": enc.key, "version": enc.version, "name": enc.name, "dims": enc.dims},
        "query": {"width": loaded.width, "height": loaded.height, "sha256": loaded.sha256,
                  "quality_warnings": loaded.quality_warnings, "garment_isolation": pq.crop_info,
                  "predicted_attributes": ctx.predicted, "palette": pq.pix["palette"],
                  "text": None if not pq.requested and pq.composition is None else (pq.composition or {}).get("target_text"),
                  "requested_attributes": pq.requested, "composition": pq.composition},
        "candidates_considered": len(cands),
        "colour_requirement": ctx.colour_requirement,
        "retrieval": info,
        "results": [_format(r) for r in ranked],
        "timings_ms": {k2: round(v, 1) for k2, v in timings.items()},
    }


def search_image(conn, loaded: LoadedImage, model_key: str | None = None, text: str | None = None,
                 top_k: int = 20, filters: dict | None = None, crop: bool = False,
                 candidates: int | None = None, ef_search: int | None = None, target=None) -> dict:
    """Public entry point (unchanged signature + optional candidates / ef_search / target garment)."""
    pq = prepare_query(loaded, model_key, text=text, crop=crop, target=target)
    res = run_query(conn, pq, top_k=top_k, filters=filters, candidates=candidates, ef_search=ef_search)
    res["query"]["text"] = text
    return res


def similar_to_product(conn, product_id: int, model_key: str | None = None, top_k: int = 12) -> dict | None:
    """'You may also like' for a product page, using its STORED embedding (no re-encoding)."""
    model_key = model_key or get_settings().ai_primary_model
    version = MODEL_SPECS[model_key]["version"]
    vec = vector_store.stored_embedding(conn, product_id, model_key, version)
    if vec is None:
        return None
    me = vector_store.product_details(conn, [product_id]).get(product_id)
    if me is None:
        return None
    k = max(top_k, get_settings().search_candidates)
    cands = vector_store.knn(conn, model_key, version, vec, k=k, exclude_product_id=product_id)
    details = vector_store.product_details(conn, [c["product_id"] for c in cands])
    attrs = me.get("attributes", {})
    predicted = {a: {"value": attrs.get(a, "unknown"), "source": "catalogue"} for a in ("pattern", "sleeve_type", "neckline")}
    predicted["category"] = {"value": me["category"], "source": "catalogue"}
    ctx = QueryContext(model_key=model_key, palette=me.get("colour_palette") or [],
                       pattern_vector=np.asarray(me["pattern_vector"]) if me.get("pattern_vector") else None,
                       predicted=predicted)
    ranked = rerank(ctx, cands, details, top_k)
    for r in ranked:  # another catalogue item is never "the same product"
        if r["match_type"] == "exact":
            r["match_type"] = "similar"
            r["reasons"] = [x for x in r["reasons"] if not x.startswith("Likely the same")]
    return {"product_id": product_id, "model": model_key, "results": [_format(r) for r in ranked]}
