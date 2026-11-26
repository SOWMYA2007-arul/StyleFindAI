"""Verify and benchmark pgvector HNSW usage (no model needed: stored embeddings are the queries).

For every model with embeddings and every k in --k and ef_search in --ef:
  * planner plan  : what PostgreSQL chooses by itself (EXPLAIN ANALYZE, node type)
  * forced index  : same query with seq/bitmap scans disabled (to test the HNSW index itself)
  * exact         : index scans disabled = ground truth
  -> recall@k of each vs exact, rows returned, latency p50/p95
Repeated with the most selective category filter, because approximate search + filter can return
FEWER rows than requested (the application then falls back to an exact scan).

    python scripts/benchmark_vector_index.py --models fashion-clip --k 50 100 200 --ef 40 64 100 200 400
Writes docs/INDEX_BENCHMARK.md and dataset/evaluation/experiments/index_benchmark_<date>.json
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
from psycopg import sql

from _common import DATASET_DIR, PROJECT_ROOT
from app.models.encoders import MODEL_SPECS
from app.services import vector_store
from app.services.db import connection


MODES = {  # every setting given explicitly each time, restored afterwards (no leaking between runs)
    "planner": {"enable_seqscan": "on", "enable_bitmapscan": "on", "enable_indexscan": "on"},
    "forced": {"enable_seqscan": "off", "enable_bitmapscan": "off", "enable_indexscan": "on"},
    "exact": {"enable_seqscan": "on", "enable_bitmapscan": "on", "enable_indexscan": "off"},
}


def _settings(mode: str, ef: int) -> dict:
    return {**MODES[mode], "hnsw.ef_search": str(ef)}


def plan_node(conn, query, params, wanted: dict) -> str:
    with vector_store.scoped_settings(conn, wanted):
        rows = conn.execute(sql.SQL("EXPLAIN (ANALYZE, FORMAT JSON) ") + query, params, prepare=False).fetchone()
    plan = rows["QUERY PLAN"][0]["Plan"] if isinstance(rows, dict) else rows[0][0]["Plan"]
    found = []

    def walk(n):
        if "Index Name" in n and "embedding" in str(n.get("Index Name", "")) or str(n.get("Index Name", "")).startswith("hnsw"):
            found.append(f"{n['Node Type']} on {n['Index Name']}")
        elif n.get("Node Type") == "Seq Scan" and n.get("Relation Name") == "image_embeddings":
            found.append("Seq Scan on image_embeddings")
        for c in n.get("Plans", []):
            walk(c)
    walk(plan)
    return ", ".join(dict.fromkeys(found)) or "exact (no HNSW; embeddings fetched per product)"


def timed(conn, query, params, wanted: dict) -> tuple[list[int], float]:
    with vector_store.scoped_settings(conn, wanted):
        t = time.perf_counter()
        rows = conn.execute(query, params, prepare=False).fetchall()  # never reuse a cached plan
        ms = (time.perf_counter() - t) * 1000
    return [r["product_id"] for r in rows], ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--k", nargs="+", type=int, default=[50, 100, 200])
    ap.add_argument("--ef", nargs="+", type=int, default=[40, 64, 100, 200, 400])
    ap.add_argument("--queries", type=int, default=50)
    args = ap.parse_args()
    out = {"generated": datetime.now(timezone.utc).isoformat(), "results": []}
    L = ["# pgvector HNSW benchmark", "", f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by "
         "`scripts/benchmark_vector_index.py`. Queries = stored catalogue embeddings (self-match excluded "
         "from nothing; exact search is ground truth).", ""]
    with connection() as conn:
        pgv = ".".join(map(str, vector_store.pgvector_version(conn)))
        L += [f"pgvector {pgv}. `planner` = PostgreSQL's own choice; `forced` = seq/bitmap scans disabled.", ""]
        models = args.models or [r["model_key"] for r in vector_store.models_with_embeddings(conn) if r["n"] > 0]
        for key in models:
            if key not in MODEL_SPECS:
                continue
            ver, dims = MODEL_SPECS[key]["version"], MODEL_SPECS[key]["dims"]
            vecs = conn.execute("""SELECT e.embedding FROM image_embeddings e WHERE model_key=%s AND model_version=%s
                                   ORDER BY e.id LIMIT %s""", (key, ver, args.queries)).fetchall()
            if not vecs:
                continue
            Q = [np.asarray(v["embedding"].to_numpy() if hasattr(v["embedding"], "to_numpy") else v["embedding"],
                            dtype=np.float32) for v in vecs]
            n_rows = conn.execute("SELECT count(*) AS n FROM image_embeddings WHERE model_key=%s AND model_version=%s",
                                  (key, ver)).fetchone()["n"]
            cat = conn.execute("""SELECT c.slug, count(*) AS n FROM products p JOIN categories c ON c.id=p.category_id
                                  WHERE p.is_active GROUP BY 1 HAVING count(*) > 0 ORDER BY 2 LIMIT 1""").fetchone()
            L += [f"## {key} ({ver}, {dims}-d, {n_rows} vectors)", "",
                  "| filter | k | ef_search | planner plan | planner recall | planner p50 ms | forced recall | forced rows (min) | forced p50 ms | exact p50 ms |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for flt in ({}, {"category": [cat["slug"]]} if cat else None):
                if flt is None:
                    continue
                q_sql, params0 = vector_store._build_knn_sql(key, dims, flt, None)
                for k in args.k:
                    exact_ids, exact_ms = [], []
                    for q in Q:
                        ids, ms = timed(conn, q_sql, {**params0, "ver": ver, "q": q, "k": k}, _settings("exact", 100))
                        exact_ids.append(ids); exact_ms.append(ms)
                    for ef in args.ef:
                        if ef < 1 or ef > 1000:
                            continue
                        rec_p, rec_f, ms_p, ms_f, rows_f = [], [], [], [], []
                        for q, truth in zip(Q, exact_ids):
                            p = {**params0, "ver": ver, "q": q, "k": k}
                            ids, ms = timed(conn, q_sql, p, _settings("planner", ef))
                            rec_p.append(len(set(ids) & set(truth)) / max(1, len(truth))); ms_p.append(ms)
                            ids, ms = timed(conn, q_sql, p, _settings("forced", ef))
                            rec_f.append(len(set(ids) & set(truth)) / max(1, len(truth))); ms_f.append(ms); rows_f.append(len(ids))
                        node = plan_node(conn, q_sql, {**params0, "ver": ver, "q": Q[0], "k": k}, _settings("planner", ef))
                        row = {"model": key, "filter": flt or None, "k": k, "ef_search": ef, "planner_plan": node,
                               "planner_recall": float(np.mean(rec_p)), "planner_p50_ms": float(np.median(ms_p)),
                               "forced_recall": float(np.mean(rec_f)), "forced_min_rows": int(min(rows_f)),
                               "forced_p50_ms": float(np.median(ms_f)), "exact_p50_ms": float(np.median(exact_ms)),
                               "vectors": n_rows, "pgvector": pgv}
                        out["results"].append(row)
                        L.append(f"| {'category=' + flt['category'][0] if flt else 'none'} | {k} | {ef} | {node} | "
                                 f"{row['planner_recall']:.3f} | {row['planner_p50_ms']:.1f} | {row['forced_recall']:.3f} | "
                                 f"{row['forced_min_rows']} | {row['forced_p50_ms']:.1f} | {row['exact_p50_ms']:.1f} |")
                        print(L[-1], flush=True)
            L.append("")
    L += ["## Reading the table", "",
          "* If `planner plan` is a Seq Scan, PostgreSQL decided an exact scan is cheaper (normal for a few thousand",
          "  vectors). Results are then exact; do not force the index.",
          "* `forced rows (min)` below k with a filter shows the classic approximate-search + filter problem. The app",
          "  then uses pgvector >= 0.8 iterative scans or falls back to an exact scan (FILTERED_EXACT_FALLBACK).",
          "* Choose the smallest ef_search whose forced recall is ~1.0 for your k; keep HNSW_EF_SEARCH=0 (auto) otherwise."]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    exp = DATASET_DIR / "evaluation" / "experiments"
    exp.mkdir(parents=True, exist_ok=True)
    (exp / f"index_benchmark_{stamp}.json").write_text(json.dumps(out, indent=2))
    (PROJECT_ROOT / "docs" / "INDEX_BENCHMARK.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
