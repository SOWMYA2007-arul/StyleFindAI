"""HTTP API of the AI service. Interactive docs: http://127.0.0.1:8001/docs"""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from pydantic import ValidationError

from app.config import get_settings
from app.models.encoders import MODEL_SPECS, ModelUnavailableError, get_registry
from app.schemas.api import IndexRequest, SearchFilters
from app.services import vector_store
from app.services.attributes import pixel_attributes, vlm_attributes, zero_shot
from app.services.db import connection
from app.services.detection import isolate_garment
from app.services.indexing import index_images, pending_images
from app.services.preprocess import ImageValidationError, load_and_validate
from app.services.search import search_image, similar_to_product
from app.utils.logging import get_logger

log = get_logger("api")
router = APIRouter()


async def _read_image(image: UploadFile):
    s = get_settings()
    max_bytes = s.max_upload_mb * 1024 * 1024
    data = await image.read(max_bytes + 1)
    try:
        return load_and_validate(data, max_bytes=max_bytes)
    except ImageValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


def _model(model: str | None) -> str:
    key = model or get_settings().ai_primary_model
    if key not in MODEL_SPECS:
        raise HTTPException(422, f"Unknown model '{key}'. Available: {list(MODEL_SPECS)}")
    return key


def _filters(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return SearchFilters.model_validate(json.loads(raw)).model_dump(exclude_none=True)
    except (json.JSONDecodeError, ValidationError, TypeError):
        raise HTTPException(422, "filters must be a JSON object like {\"category\": [\"sarees\"]}") from None


def _target(raw: str | None):
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()[:40]
    if raw.isdigit():
        if int(raw) > 20:
            raise HTTPException(422, "target index must be between 0 and 20")
        return int(raw)
    if not all(c.isalnum() or c in " -_" for c in raw):
        raise HTTPException(422, "target must be an instance index or a garment word like 'upper' or 'dupatta'")
    return raw.lower()


def _require_internal(token: str | None) -> None:
    if token != get_settings().ai_internal_token:
        raise HTTPException(401, "missing or invalid internal token")


@router.get("/health")
def health():
    s = get_settings()
    out = {"status": "ok", "primary_model": s.ai_primary_model, "loaded_models": get_registry().loaded_keys(),
           "device": s.ai_device}
    try:
        with connection() as conn:
            v = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
            out["database"] = "ok"
            out["pgvector"] = v["extversion"] if v else None
            out["products"] = conn.execute("SELECT count(*) AS n FROM products WHERE is_active").fetchone()["n"]
            out["embeddings"] = {f"{r['model_key']}": r["n"] for r in vector_store.models_with_embeddings(conn)}
            from app.services.indexing import FEATURE_VERSION

            stale = conn.execute("""SELECT count(*) AS n FROM product_images pi LEFT JOIN image_features f
                                    ON f.product_image_id = pi.id
                                    WHERE f.product_image_id IS NULL OR f.feature_version <> %s""",
                                 (FEATURE_VERSION,)).fetchone()["n"]
            out["outdated_features"] = stale
            if stale:
                out["warnings"] = [f"{stale} catalogue images have outdated colour/pattern features; "
                                   "run: python scripts/recompute_features.py"]
    except Exception as e:  # noqa: BLE001
        log.error("health db check failed: %s", e)
        out.update(status="degraded", database="unavailable")
    return out


@router.post("/ai/embed")
async def embed(image: UploadFile = File(...), model: str | None = Form(None)):
    key = _model(model)
    loaded = await _read_image(image)
    enc = get_registry().get(key)
    t = time.perf_counter()
    vec = enc.embed_images([loaded.image])[0]
    return {"model": key, "model_version": enc.version, "dims": enc.dims,
            "embedding": [round(float(x), 6) for x in vec], "embed_ms": round((time.perf_counter() - t) * 1000, 1)}


@router.post("/ai/search")
async def search(image: UploadFile = File(...), text: str | None = Form(None), model: str | None = Form(None),
                 top_k: int = Form(20), filters: str | None = Form(None), crop: bool = Form(False),
                 candidates: int | None = Form(None), ef_search: int | None = Form(None),
                 target: str | None = Form(None)):
    """target (optional, with crop=true): which garment to search for when the photo has several -
    an instance index from /ai/segment ("0", "1"...) or a group/label ("upper", "lower", "dress", "dupatta")."""
    t0 = time.perf_counter()
    key = _model(model)
    if not 1 <= top_k <= 50:
        raise HTTPException(422, "top_k must be between 1 and 50")
    if candidates is not None and not top_k <= candidates <= 500:
        raise HTTPException(422, "candidates must be between top_k and 500")
    if ef_search is not None and not 1 <= ef_search <= 1000:
        raise HTTPException(422, "ef_search must be between 1 and 1000")
    if text is not None:
        text = text.strip()[:300] or None
    flt = _filters(filters)
    tgt = _target(target)
    loaded = await _read_image(image)
    preprocess_ms = (time.perf_counter() - t0) * 1000
    with connection() as conn:
        n = conn.execute("SELECT count(*) AS n FROM image_embeddings WHERE model_key=%s AND model_version=%s",
                         (key, MODEL_SPECS[key]["version"])).fetchone()["n"]
        if n == 0:
            raise HTTPException(409, f"No catalogue embeddings for model '{key}'. "
                                     f"Run: python scripts/generate_embeddings.py --model {key}")
        try:
            res = search_image(conn, loaded, key, text=text, top_k=top_k, filters=flt, crop=crop,
                               candidates=candidates, ef_search=ef_search, target=tgt)
        except ModelUnavailableError as e:
            raise HTTPException(503, str(e)) from None
    res["timings_ms"]["preprocess_ms"] = round(preprocess_ms, 1)
    res["timings_ms"]["request_total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return res


@router.post("/ai/extract-attributes")
async def extract_attributes(image: UploadFile = File(...), model: str | None = Form(None),
                             crop: bool = Form(False)):
    key = _model(model)
    loaded = await _read_image(image)
    img, info = isolate_garment(loaded.image, loaded.sha256) if crop else (loaded.image, {"method": "full_image"})
    pix = pixel_attributes(img)
    attrs = {"colour": pix["colour"], "secondary_colour": pix["secondary_colour"], "texture": pix["texture"]}
    enc = get_registry().get(key)
    if enc.supports_text:
        attrs.update(zero_shot(enc, enc.embed_images([img])[0]))
    attrs.update(vlm_attributes(img))
    return {"model": key, "garment_isolation": info, "attributes": attrs, "palette": pix["palette"],
            "note": "Predicted attributes of the uploaded image; 'unknown' means not confident enough."}


@router.post("/ai/segment")
async def segment_image(image: UploadFile = File(...), target: str | None = Form(None),
                        strategy: str | None = Form(None), include_masks: bool = Form(False)):
    """Garment segmentation only (no search): route taken, every garment instance (label, group, confidence,
    bbox, occlusion, optional PNG mask as base64), the chosen target, fallbacks and per-stage timings.
    strategy: auto (default, uses models enabled in .env) | light (weight-free only) | off."""
    from app.services.segmentation.pipeline import segment

    if strategy is not None and strategy not in ("auto", "light", "off"):
        raise HTTPException(422, "strategy must be auto, light or off")
    tgt = _target(target)
    loaded = await _read_image(image)
    t = time.perf_counter()
    try:
        res = segment(loaded.image, loaded.sha256, tgt, strategy)
    except Exception as e:  # noqa: BLE001
        log.exception("segmentation failed")
        raise HTTPException(500, f"segmentation failed: {type(e).__name__}") from None
    out = res.to_dict(include_masks=include_masks)
    out["request_ms"] = round((time.perf_counter() - t) * 1000, 1)
    return out


@router.get("/ai/segmentation/status")
def segmentation_status():
    """Which segmentation backends are configured / installed / loaded (no model is loaded by this call)."""
    from app.services.segmentation import backends as B
    from app.services.segmentation.pipeline import cache

    s = get_settings()
    return {"garment_isolation": s.garment_isolation, "strategy": s.segmentation_strategy,
            "backends": B.status(), "mask_cache": cache().stats(),
            "min_segmentation_confidence": s.min_segmentation_confidence, "device": s.ai_device}


@router.post("/ai/compare-models")
async def compare_models(image: UploadFile = File(...), top_k: int = Form(10)):
    loaded = await _read_image(image)
    out = {}
    with connection() as conn:
        available = [r for r in vector_store.models_with_embeddings(conn) if r["n"] > 0]
        for r in available:
            try:
                res = search_image(conn, loaded, r["model_key"], top_k=min(top_k, 20))
                out[r["model_key"]] = {"timings_ms": res["timings_ms"],
                                       "results": [{k: x[k] for k in ("product_id", "product_code", "name", "category",
                                                                       "image_key", "scores", "match_type")}
                                                   for x in res["results"]]}
            except Exception as e:  # noqa: BLE001
                log.warning("compare: %s failed: %s", r["model_key"], e)
                out[r["model_key"]] = {"error": "model unavailable"}
    return {"models_compared": list(out), "comparison": out}


@router.get("/ai/products/{product_id}/similar")
def similar(product_id: int, top_k: int = 12, model: str | None = None):
    key = _model(model)
    with connection() as conn:
        res = similar_to_product(conn, product_id, key, top_k=max(1, min(top_k, 30)))
    if res is None:
        raise HTTPException(404, "product not found or has no embedding for this model")
    return res


@router.post("/ai/index")
def index(req: IndexRequest, x_internal_token: str | None = Header(None)):
    """Admin-only (called by the Node backend): embed new / changed products."""
    _require_internal(x_internal_token)
    key = _model(req.model)
    enc = get_registry().get(key)
    with connection() as conn:
        rows = pending_images(conn, enc, product_ids=req.product_ids or None, force=req.force)
        return index_images(conn, enc, rows)


@router.get("/ai/models")
def models():
    with connection() as conn:
        counts = {r["model_key"]: r["n"] for r in vector_store.models_with_embeddings(conn)}
    return [{"key": k, "name": v["name"], "dims": v["dims"], "version": v["version"], "supports_text": v["text"],
             "catalogue_embeddings": counts.get(k, 0)} for k, v in MODEL_SPECS.items()]
