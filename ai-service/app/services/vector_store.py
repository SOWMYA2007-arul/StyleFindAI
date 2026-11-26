"""pgvector retrieval.

`<=>` is pgvector's cosine-DISTANCE operator (1 - cosine similarity). We cast the untyped
`embedding` column to the model's dimension so PostgreSQL can use that model's partial HNSW
index (see database/migrations/002_vector_indexes.sql).
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
from psycopg import sql

from app.models.encoders import MODEL_SPECS


def _validated(model_key: str) -> int:
    if model_key not in MODEL_SPECS:  # whitelist -> safe to inline as a literal
        raise KeyError(model_key)
    return int(MODEL_SPECS[model_key]["dims"])


_PGVECTOR_VERSION: tuple[int, ...] | None = None


def pgvector_version(conn) -> tuple[int, ...]:
    """(major, minor, patch) of the installed pgvector extension, cached per process."""
    global _PGVECTOR_VERSION
    if _PGVECTOR_VERSION is None:
        r = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        v = (r["extversion"] if isinstance(r, dict) else r[0]) if r else "0"
        _PGVECTOR_VERSION = tuple(int(x) for x in v.split(".") if x.isdigit())
    return _PGVECTOR_VERSION


@contextmanager
def scoped_settings(conn, wanted: dict[str, str]):
    """Apply planner settings for ONE query and restore the previous values afterwards.

    Why not just `SET LOCAL` inside conn.transaction()? If a transaction is already open (the
    usual case inside a request), psycopg turns `transaction()` into a SAVEPOINT, and PostgreSQL
    keeps SET LOCAL values until the OUTER transaction ends - they would leak into later queries
    (e.g. an exact-scan fallback silently disabling the index for the rest of the request).
    """
    with conn.transaction():
        old = {}
        for k, v in wanted.items():
            row = conn.execute("SELECT current_setting(%s, true) AS v", (k,), prepare=False).fetchone()
            old[k] = row["v"] if isinstance(row, dict) else row[0]
            conn.execute("SELECT set_config(%s, %s, true)", (k, v), prepare=False)
        try:
            yield
        finally:
            for k, v in old.items():
                if v is not None:
                    conn.execute("SELECT set_config(%s, %s, true)", (k, v), prepare=False)


def resolve_ef_search(k: int, override: int | None = None) -> int:
    """HNSW returns at most ef_search rows, so it must be >= k. pgvector allows 1..1000."""
    from app.config import get_settings

    ef = override if override else get_settings().hnsw_ef_search
    if not ef:
        ef = max(100, 2 * k)
    return int(min(1000, max(k, ef)))


def _build_knn_sql(model_key: str, dims: int, filters: dict, exclude_product_id: int | None):
    where = [sql.SQL("e.model_key = {}").format(sql.Literal(model_key)),
             sql.SQL("e.model_version = %(ver)s"), sql.SQL("p.is_active")]
    params: dict = {}
    if filters.get("category"):
        where.append(sql.SQL("c.slug = ANY(%(cats)s)")); params["cats"] = list(filters["category"])
    for attr in ("colour", "pattern", "sleeve_type", "neckline"):
        vals = filters.get(attr)
        if vals:
            # Only VERIFIED catalogue attributes may filter results (never model predictions).
            where.append(sql.SQL(
                "EXISTS (SELECT 1 FROM product_attributes pa WHERE pa.product_id = p.id AND pa.verified "
                "AND pa.attribute = {a} AND pa.value = ANY({v}))").format(a=sql.Literal(attr), v=sql.Placeholder(f"f_{attr}")))
            params[f"f_{attr}"] = [str(v).lower() for v in vals]
    if filters.get("colour_evidence"):
        # INTERNAL (not a shopper filter): candidates for a colour the shopper typed. Any colour evidence counts
        # here; the reranker then applies the strict evidence order (catalogue > measured garment > photo).
        where.append(sql.SQL(
            "(EXISTS (SELECT 1 FROM product_attributes pa WHERE pa.product_id = p.id AND pa.attribute = 'colour' "
            "AND (pa.verified OR pa.source = 'model_prediction') AND pa.value = ANY(%(ce)s)) "
            "OR EXISTS (SELECT 1 FROM image_features f WHERE f.product_image_id = pi.id AND f.colour_name = ANY(%(ce)s)))"))
        params["ce"] = sorted(filters["colour_evidence"])
    if filters.get("min_price") is not None:
        where.append(sql.SQL("p.price >= %(min_price)s")); params["min_price"] = float(filters["min_price"])
    if filters.get("max_price") is not None:
        where.append(sql.SQL("p.price <= %(max_price)s")); params["max_price"] = float(filters["max_price"])
    if exclude_product_id is not None:
        where.append(sql.SQL("p.id <> %(excl)s")); params["excl"] = exclude_product_id
    cast = sql.SQL("vector({})").format(sql.Literal(dims))
    q = sql.SQL("""
        SELECT p.id AS product_id, pi.id AS image_id,
               1 - (e.embedding::{cast} <=> %(q)s::{cast}) AS cosine
        FROM image_embeddings e
        JOIN product_images pi ON pi.id = e.product_image_id AND pi.is_primary
        JOIN products p ON p.id = pi.product_id
        JOIN categories c ON c.id = p.category_id
        WHERE {where}
        ORDER BY e.embedding::{cast} <=> %(q)s::{cast}
        LIMIT %(k)s""").format(cast=cast, where=sql.SQL(" AND ").join(where))
    return q, params


def knn(conn, model_key: str, model_version: str, query: np.ndarray, k: int = 50,
        filters: dict | None = None, exclude_product_id: int | None = None,
        ef_search: int | None = None, exact: bool = False, info: dict | None = None) -> list[dict]:
    """Top-k catalogue images by cosine similarity for ONE model/version.

    * Inactive products and non-primary images are never returned.
    * `info` (optional dict) is filled with how the query was answered, for timing/debugging.
    * Results are tie-broken by product id so repeated searches give the same order.
    """
    dims = _validated(model_key)
    q = np.asarray(query, dtype=np.float32)
    if q.shape != (dims,):
        raise ValueError(f"query has shape {q.shape}, model {model_key} expects ({dims},)")
    if not np.all(np.isfinite(q)):
        raise ValueError("query embedding contains NaN/inf")
    from app.config import get_settings

    s = get_settings()
    filters = {k2: v for k2, v in (filters or {}).items() if v not in (None, [], "")}
    filtered = bool(filters) or exclude_product_id is not None
    query_sql, params = _build_knn_sql(model_key, dims, filters, exclude_product_id)
    params.update({"ver": model_version, "q": q, "k": k})
    ef = resolve_ef_search(k, ef_search)
    iterative = filtered and s.hnsw_iterative_scan and pgvector_version(conn) >= (0, 8, 0)

    def run(exact_scan: bool) -> list[dict]:
        wanted = {"hnsw.ef_search": str(ef), "enable_indexscan": "off" if exact_scan else "on"}
        if pgvector_version(conn) >= (0, 8, 0):
            # strict_order keeps exact distance ordering while scanning further for rows that pass filters
            wanted["hnsw.iterative_scan"] = "strict_order" if (iterative and not exact_scan) else "off"
        with scoped_settings(conn, wanted):
            # prepare=False: never let psycopg turn this into a server-side prepared statement whose
            # cached plan would ignore the settings above / the actual filter values.
            return list(conn.execute(query_sql, params, prepare=False))

    rows = run(exact)
    mode = "exact" if exact else "planner"
    if not exact and filtered and len(rows) < k and s.filtered_exact_fallback:
        exact_rows = run(True)
        if len(exact_rows) > len(rows):
            rows, mode = exact_rows, "exact_fallback"
    rows.sort(key=lambda r: (-round(float(r["cosine"]), 6), r["product_id"]))
    if info is not None:
        info.update({"mode": mode, "ef_search": ef, "iterative_scan": iterative, "requested": k,
                     "returned": len(rows), "filtered": filtered})
    return rows


def product_details(conn, product_ids: list[int]) -> dict[int, dict]:
    if not product_ids:
        return {}
    rows = conn.execute("""
        SELECT p.id, p.product_code, p.name, p.subcategory, p.price, c.slug AS category, c.name AS category_name,
               pi.id AS image_id, pi.storage_key, pi.phash,
               f.colour_palette, f.colour_name AS pixel_colour, f.pattern_vector, f.pattern_label
        FROM products p
        JOIN categories c ON c.id = p.category_id
        JOIN product_images pi ON pi.product_id = p.id AND pi.is_primary
        LEFT JOIN image_features f ON f.product_image_id = pi.id
        WHERE p.id = ANY(%s)""", (product_ids,)).fetchall()
    out = {r["id"]: dict(r, attributes={}) for r in rows}
    for a in conn.execute("""SELECT product_id, attribute, value, source FROM product_attributes
                             WHERE product_id = ANY(%s) AND verified ORDER BY source""", (product_ids,)):
        out[a["product_id"]]["attributes"].setdefault(a["attribute"], a["value"])
    # measured colour (model_prediction, e.g. from the dataset importers). Never shown or used by filters as a
    # catalogue fact; only used to judge a colour the shopper typed ("... in green") better than the
    # whole-photo pixel colour, which a coloured wall or floor can distort.
    for a in conn.execute("""SELECT product_id, value, confidence FROM product_attributes
                             WHERE product_id = ANY(%s) AND attribute = 'colour' AND NOT verified
                               AND source = 'model_prediction' AND value <> 'unknown'""", (product_ids,)):
        out[a["product_id"]]["measured_colour"] = a["value"]
    return out


def stored_embedding(conn, product_id: int, model_key: str, model_version: str) -> np.ndarray | None:
    _validated(model_key)
    r = conn.execute("""SELECT e.embedding FROM image_embeddings e
                        JOIN product_images pi ON pi.id = e.product_image_id
                        JOIN products p ON p.id = pi.product_id AND p.is_active
                        WHERE pi.product_id = %s AND pi.is_primary AND e.model_key = %s AND e.model_version = %s""",
                     (product_id, model_key, model_version)).fetchone()
    if r is None:
        return None
    v = r["embedding"]
    return np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v, dtype=np.float32)


def models_with_embeddings(conn) -> list[dict]:
    return list(conn.execute("""SELECT m.model_key, m.model_version, m.dims, count(e.id) AS n
                                FROM embedding_models m LEFT JOIN image_embeddings e
                                  ON e.model_key = m.model_key AND e.model_version = m.model_version
                                GROUP BY 1,2,3 ORDER BY 1"""))
