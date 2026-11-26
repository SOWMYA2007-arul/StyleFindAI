"""Offline catalogue indexing: embeddings + colour/pattern features for product images.

Used by scripts/generate_embeddings.py (bulk) and by the admin endpoint (single product).
Catalogue embeddings are computed ONCE and stored; user searches only embed the query.
"""
from __future__ import annotations

import json
import time

import numpy as np
from PIL import Image

from app.config import get_settings
from app.models.encoders import Encoder
from app.services.colour import PALETTE_VERSION, extract_palette
from app.services.foreground import FOREGROUND_VERSION, estimate_foreground
from app.services.pattern import FEATURE_VERSION as PATTERN_DESC_VERSION
from app.services.pattern import PATTERN_CLASSIFIER_VERSION, classify_pattern, pattern_descriptor

# Catalogue colour/pattern features must be computed exactly like query features.
FEATURE_VERSION = f"{FOREGROUND_VERSION}|{PALETTE_VERSION}|{PATTERN_DESC_VERSION}|{PATTERN_CLASSIFIER_VERSION}"
from app.services.preprocess import to_rgb
from app.utils.logging import get_logger

log = get_logger("indexing")


def ensure_model_row(conn, enc: Encoder) -> None:
    conn.execute(
        """INSERT INTO embedding_models (model_key, model_version, model_name, dims)
           VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING""",
        (enc.key, enc.version, enc.name, enc.dims))
    row = conn.execute("SELECT dims FROM embedding_models WHERE model_key=%s AND model_version=%s",
                       (enc.key, enc.version)).fetchone()
    if row["dims"] != enc.dims:
        raise RuntimeError(f"DB says {enc.key} has {row['dims']} dims but encoder produces {enc.dims}")


def pending_images(conn, enc: Encoder, product_ids: list[int] | None = None, force: bool = False) -> list[dict]:
    """Images that have no embedding for this exact model version (or all, if force)."""
    sql = """SELECT pi.id, pi.storage_key FROM product_images pi
             JOIN products p ON p.id = pi.product_id
             WHERE p.is_active"""
    params: list = []
    if product_ids:
        sql += " AND p.id = ANY(%s)"
        params.append(product_ids)
    if not force:
        sql += """ AND NOT EXISTS (SELECT 1 FROM image_embeddings e WHERE e.product_image_id = pi.id
                                   AND e.model_key = %s AND e.model_version = %s)"""
        params += [enc.key, enc.version]
    return list(conn.execute(sql + " ORDER BY pi.id", params))


def _load(storage_key: str) -> Image.Image:
    with Image.open(get_settings().products_path / storage_key) as im:
        return to_rgb(im)


def _mark_failed(conn, image_id: int, error: str) -> None:
    """Best effort: record the failure for the admin panel (needs migration 003)."""
    try:
        with conn.transaction():
            conn.execute("UPDATE product_images SET index_status = 'failed', index_error = %s WHERE id = %s",
                         (error[:500], image_id))
    except Exception:  # noqa: BLE001 - column missing on an old schema
        pass


def index_images(conn, enc: Encoder, rows: list[dict], batch_size: int | None = None,
                 with_features: bool = True, progress: bool = False) -> dict:
    """Embed images in batches. A broken image is marked 'failed' and skipped; it never stops the batch."""
    ensure_model_row(conn, enc)
    bs = batch_size or get_settings().ai_batch_size
    done, failed, t0 = 0, [], time.perf_counter()
    for i in range(0, len(rows), bs):
        batch, images = [], []
        for r in rows[i:i + bs]:
            try:
                images.append(_load(r["storage_key"])); batch.append(r)
            except Exception as e:  # noqa: BLE001 - missing / corrupt file
                log.warning("cannot load image %s: %s", r["id"], e)
                _mark_failed(conn, r["id"], f"image could not be read ({type(e).__name__})")
                failed.append(r["id"])
        if not batch:
            continue
        try:
            vecs = enc.embed_images(images)
            assert vecs.shape == (len(batch), enc.dims), f"bad embedding shape {vecs.shape}"
            for r, v, im in zip(batch, vecs, images):
                conn.execute(
                    """INSERT INTO image_embeddings (product_image_id, model_key, model_version, dims, embedding)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (product_image_id, model_key, model_version)
                       DO UPDATE SET embedding = EXCLUDED.embedding, created_at = now()""",
                    (r["id"], enc.key, enc.version, enc.dims, np.asarray(v, dtype=np.float32)))
                if with_features:
                    upsert_features(conn, r["id"], im)
            conn.commit()
            done += len(batch)
        except Exception as e:  # noqa: BLE001
            conn.rollback()
            log.error("embedding batch failed: %s", e)
            for r in batch:
                _mark_failed(conn, r["id"], f"embedding failed ({type(e).__name__})")
                failed.append(r["id"])
            conn.commit()
        if progress:
            rate = done / max(1e-6, time.perf_counter() - t0)
            print(f"  {done}/{len(rows)} images  ({rate:.1f} img/s, {len(failed)} failed)", flush=True)
    return {"model_key": enc.key, "model_version": enc.version, "indexed": done, "failed": len(failed),
            "failed_image_ids": failed[:50], "seconds": round(time.perf_counter() - t0, 2)}


def upsert_features(conn, image_id: int, im: Image.Image) -> None:
    fg = estimate_foreground(im, version="v2")  # must match FOREGROUND_VERSION (fg-v2)
    palette = extract_palette(im, fg=fg)
    pvec, _ = pattern_descriptor(im, fg=fg)
    pattern = classify_pattern(im, fg=fg)
    conn.execute(
        """INSERT INTO image_features (product_image_id, feature_version, colour_palette, colour_name,
                                       pattern_vector, pattern_label)
           VALUES (%s, %s, %s, %s, %s, %s)
           ON CONFLICT (product_image_id) DO UPDATE SET feature_version = EXCLUDED.feature_version,
             colour_palette = EXCLUDED.colour_palette, colour_name = EXCLUDED.colour_name,
             pattern_vector = EXCLUDED.pattern_vector, pattern_label = EXCLUDED.pattern_label, updated_at = now()""",
        (image_id, FEATURE_VERSION, json.dumps(palette), palette[0]["name"] if palette else None,
         [float(x) for x in pvec], pattern["value"]))


def stale_feature_images(conn, force: bool = False) -> list[dict]:
    sql = """SELECT pi.id, pi.storage_key FROM product_images pi
             LEFT JOIN image_features f ON f.product_image_id = pi.id"""
    if not force:
        sql += " WHERE f.product_image_id IS NULL OR f.feature_version <> %s"
        return list(conn.execute(sql + " ORDER BY pi.id", (FEATURE_VERSION,)))
    return list(conn.execute(sql + " ORDER BY pi.id"))


def recompute_features(conn, rows: list[dict], progress: bool = False) -> int:
    done = 0
    for r in rows:
        upsert_features(conn, r["id"], _load(r["storage_key"]))
        done += 1
        if done % 100 == 0:
            conn.commit()
            if progress:
                print(f"  {done}/{len(rows)}", flush=True)
    conn.commit()
    return done
