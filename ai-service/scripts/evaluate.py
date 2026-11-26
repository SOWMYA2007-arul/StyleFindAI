"""Retrieval evaluation with stored, comparable experiments.

    # baseline (current settings)
    python scripts/evaluate.py --models fashion-clip --tag baseline
    # candidate-count experiment: SAME queries, model, DB and weights; only N changes
    python scripts/evaluate.py --models fashion-clip --candidates 50 100 200 --tag candidates
    # HNSW ef_search experiment
    python scripts/evaluate.py --models fashion-clip --candidates 100 --ef-search 40 100 200 --tag ef
    # garment isolation / multi-crop experiments
    python scripts/evaluate.py --models fashion-clip --crop --tag crop
    python scripts/evaluate.py --models fashion-clip --multicrop max_score --tag multicrop
    # then compare everything
    python scripts/compare_experiments.py

For every query the image is embedded ONCE per model; each candidate/ef configuration then reuses
that embedding, so configurations differ only in what is being tested.

Per configuration:
  visual  : rank of the true product in pgvector's cosine top-N (N = --candidates)
  hybrid  : rank after the hybrid re-ranker over the SAME N candidates (the website pipeline)
Metrics: Recall@1/5/10, mAP (one relevant product per query -> AP = 1/rank, 0 if not in top N),
category P@10, colour C@10, pattern C@10, exact-match precision/recall, latency p50/p95, RSS memory.

Each configuration is stored in dataset/evaluation/experiments/<id>/ (result.json + report.md) and
appended to experiments/index.jsonl. Nothing is overwritten; docs/EVALUATION.md is only rewritten
with --update-docs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
import zlib
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from _common import CATALOGUE_CSV, DATASET_DIR, PROJECT_ROOT
from app.config import get_settings
from app.models.encoders import MODEL_SPECS, get_registry
from app.services import vector_store
from app.services.db import connection
from app.services.preprocess import load_and_validate
from app.services.ranking import weights as current_weights
from app.services.search import prepare_query, run_query

EXPERIMENTS_DIR = DATASET_DIR / "evaluation" / "experiments"


# ----------------------------------------------------------------------------- helpers
def recall(ranks, k):
    return float(np.mean([r is not None and r <= k for r in ranks])) if ranks else 0.0


def mean_ap(ranks):
    return float(np.mean([1.0 / r if r else 0.0 for r in ranks])) if ranks else 0.0


def query_split(row: dict) -> str:
    """Product-level split (all queries of one product land in the same split), deterministic.
    Uses the CSV's 'split' column when present."""
    if row.get("split") in ("val", "test"):
        return row["split"]
    return "val" if zlib.crc32(row["product_id"].encode()) % 2 == 0 else "test"


def rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 1)
    except Exception:  # noqa: BLE001 - psutil optional
        return None


def file_hash(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def dataset_version(conn) -> dict:
    rows = conn.execute("""SELECT p.product_code, pi.sha256 FROM products p
                           JOIN product_images pi ON pi.product_id = p.id AND pi.is_primary
                           WHERE p.is_active ORDER BY p.product_code""").fetchall()
    h = hashlib.sha256("\n".join(f"{r['product_code']}:{r['sha256']}" for r in rows).encode()).hexdigest()[:12]
    return {"active_products": len(rows), "hash": h}


def hardware() -> dict:
    try:
        import torch

        threads = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        threads = None
    ram = None
    try:
        import psutil

        ram = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:  # noqa: BLE001
        pass
    return {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(), "torch_threads": threads, "ram_gb": ram, "device": get_settings().ai_device,
            "python": platform.python_version()}


def summarise(per: list[dict]) -> dict:
    def block(rows):
        vr = [r["visual_rank"] for r in rows]; hr = [r["hybrid_rank"] for r in rows]
        lat = [r["latency_ms"] for r in rows]
        pat = [r["pattern_c10"] for r in rows if r["pattern_c10"] is not None]
        ex = [r for r in rows if r["exact_label"]]
        return {
            "n": len(rows),
            "visual": {"R@1": recall(vr, 1), "R@5": recall(vr, 5), "R@10": recall(vr, 10), "mAP": mean_ap(vr)},
            "hybrid": {"R@1": recall(hr, 1), "R@5": recall(hr, 5), "R@10": recall(hr, 10), "mAP": mean_ap(hr)},
            "candidate_recall": float(np.mean([r["hybrid_rank"] is not None for r in rows])) if rows else 0.0,
            "category_P@10": float(np.mean([r["cat_p10"] for r in rows])) if rows else 0.0,
            "colour_C@10": float(np.mean([r["colour_c10"] for r in rows])) if rows else 0.0,
            "pattern_C@10": float(np.mean(pat)) if pat else None,
            "exact_labelled": len(ex),
            "exact_precision": float(np.mean([r["exact_label"] == r["truth"] for r in ex])) if ex else None,
            "exact_recall": len([r for r in ex if r["exact_label"] == r["truth"]]) / max(1, len(rows)),
            "latency_ms_p50": float(np.percentile(lat, 50)) if lat else None,
            "latency_ms_p95": float(np.percentile(lat, 95)) if lat else None,
            "vector_search_ms_p50": float(np.percentile([r["vector_ms"] for r in rows], 50)) if rows else None,
            "exact_fallbacks": sum(1 for r in rows if r.get("retrieval_mode") == "exact_fallback"),
        }
    by_t, by_s = defaultdict(list), defaultdict(list)
    for r in per:
        by_t[r["transform"]].append(r); by_s[r["split"]].append(r)
    return {"overall": block(per), "by_transform": {t: block(rs) for t, rs in sorted(by_t.items())},
            "by_split": {s: block(rs) for s, rs in sorted(by_s.items())}}


def to_markdown(res: dict) -> str:
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    m, o = res["meta"], res["overall"]
    L = [f"# Experiment {m['id']}", "",
         f"* Date: {m['generated']}", f"* Model: `{m['model_key']}` version `{m['model_version']}`",
         f"* Candidates: {m['candidates']}, hnsw.ef_search: {m['ef_search']}, crop: {m['crop']}, multicrop: {m['multicrop']}",
         f"* Ranking weights: {m['weights']}",
         f"* Dataset: {m['dataset']['active_products']} active products, version `{m['dataset']['hash']}`, "
         f"sources {m['sources']}",
         f"* Queries: {m['n_queries']} from `{m['queries_file']}` (version `{m['query_set_version']}`, split `{m['split']}`)",
         f"* Hardware: {m['hardware']}", f"* RSS memory after run: {m['rss_mb']} MB", ""]
    if any("SYNTHETIC" in s for s in m["sources"]):
        L += ["> **Synthetic fixture.** These numbers only show that the pipeline works.", ""]
    if m.get("derived_queries", True):
        L += ["> Queries are augmented catalogue photos (known answers). They are NOT real customer photos and",
              "> must not be reported as real-world accuracy.", ""]
    L += ["| ranking | R@1 | R@5 | R@10 | mAP |", "|---|---|---|---|---|"]
    for mode in ("visual", "hybrid"):
        x = o[mode]
        L.append(f"| {mode} | {pct(x['R@1'])} | {pct(x['R@5'])} | {pct(x['R@10'])} | {x['mAP']:.3f} |")
    L += ["", f"Candidate recall (true product inside the N candidates): {pct(o['candidate_recall'])}  ",
          f"Category P@10 {pct(o['category_P@10'])}, colour C@10 {pct(o['colour_C@10'])}, pattern C@10 {pct(o['pattern_C@10'])}  ",
          f"Exact label: {o['exact_labelled']} queries, precision {pct(o['exact_precision'])}, recall {pct(o['exact_recall'])}  ",
          f"Latency p50 {o['latency_ms_p50']:.0f} ms, p95 {o['latency_ms_p95']:.0f} ms (vector search p50 "
          f"{o['vector_search_ms_p50']:.1f} ms); exact-scan fallbacks: {o['exact_fallbacks']}", "",
          "## By scenario (hybrid)", "", "| scenario | n | R@1 | R@10 | mAP | visual R@1 |", "|---|---|---|---|---|---|"]
    for t, b in res["by_transform"].items():
        L.append(f"| {t} | {b['n']} | {pct(b['hybrid']['R@1'])} | {pct(b['hybrid']['R@10'])} | {b['hybrid']['mAP']:.3f} | "
                 f"{pct(b['visual']['R@1'])} |")
    fails = [q for q in res["per_query"] if not q["hybrid_rank"] or q["hybrid_rank"] > 10][:10]
    L += ["", "## Failure cases (true product not in hybrid top 10)", ""]
    L += [f"* `{f['query']}` -> top-1 `{f['top1']}`, visual rank {f['visual_rank']}, hybrid rank {f['hybrid_rank']}"
          for f in fails] or ["* none"]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- main loop
def evaluate_model(conn, key: str, queries: list[dict], catalogue: dict, configs: list[tuple[int, int | None]],
                   crop: bool, multicrop: str, weights: dict | None) -> dict[tuple, list[dict]]:
    enc = get_registry().get(key)
    from PIL import Image

    enc.embed_images([Image.new("RGB", (224, 224))])  # warm-up so model loading is not timed
    per: dict[tuple, list[dict]] = {c: [] for c in configs}
    for i, q in enumerate(queries):
        data = (DATASET_DIR / "evaluation" / q["query_path"]).read_bytes()
        t0 = time.perf_counter()
        loaded = load_and_validate(data, max_bytes=20 * 1024 * 1024)
        pq = prepare_query(loaded, key, crop=crop, multicrop=multicrop)
        prep_ms = (time.perf_counter() - t0) * 1000
        truth = q["product_id"]
        t_meta = catalogue.get(truth, {})
        for cand, ef in configs:
            info: dict = {}
            t = time.perf_counter()
            vis = vector_store.knn(conn, enc.key, enc.version, pq.embedding, k=cand, ef_search=ef, info=info)
            vector_ms = (time.perf_counter() - t) * 1000
            codes = vector_store.product_details(conn, [c["product_id"] for c in vis])
            vis_codes = [codes[c["product_id"]]["product_code"] for c in vis if c["product_id"] in codes]
            t = time.perf_counter()
            res = run_query(conn, pq, top_k=cand, candidates=cand, ef_search=ef, weights=weights)
            run_ms = (time.perf_counter() - t) * 1000
            hyb = [r["product_code"] for r in res["results"]]
            top10 = res["results"][:10]
            exact = [r for r in res["results"] if r["match_type"] == "exact"]
            per[(cand, ef)].append({
                "query": q["query_path"], "transform": q["transform"], "split": query_split(q), "truth": truth,
                "visual_rank": vis_codes.index(truth) + 1 if truth in vis_codes else None,
                "hybrid_rank": hyb.index(truth) + 1 if truth in hyb else None,
                "top1": hyb[0] if hyb else None, "top10": hyb[:10],
                "cat_p10": float(np.mean([r["category"] == t_meta.get("category") for r in top10])) if top10 else 0.0,
                "colour_c10": float(np.mean([r["colour"] == (t_meta.get("colour") or "").lower() for r in top10])) if top10 else 0.0,
                "pattern_c10": (float(np.mean([r["pattern"] == t_meta.get("pattern") for r in top10]))
                                if top10 and t_meta.get("pattern", "unknown") != "unknown" else None),
                "exact_label": exact[0]["product_code"] if exact else None,
                "latency_ms": prep_ms + run_ms, "vector_ms": vector_ms,
                "retrieval_mode": res.get("retrieval", {}).get("mode"),
            })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(queries)} queries", flush=True)
    return per


def stability_check(conn, key: str, queries: list[dict], candidates: int, n: int = 10) -> bool:
    """Same image twice -> identical top-10 order."""
    for q in queries[:n]:
        data = (DATASET_DIR / "evaluation" / q["query_path"]).read_bytes()
        a = [r["product_code"] for r in run_query(conn, prepare_query(load_and_validate(data, 20 << 20), key),
                                                   top_k=10, candidates=candidates)["results"]]
        b = [r["product_code"] for r in run_query(conn, prepare_query(load_and_validate(data, 20 << 20), key),
                                                   top_k=10, candidates=candidates)["results"]]
        if a != b:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--queries", default=str(DATASET_DIR / "evaluation" / "queries.csv"))
    ap.add_argument("--candidates", nargs="+", type=int, default=None,
                    help="candidate counts to compare (default: SEARCH_CANDIDATES)")
    ap.add_argument("--ef-search", nargs="+", type=int, default=None, help="hnsw.ef_search values (default: automatic)")
    ap.add_argument("--split", choices=["all", "val", "test"], default="all")
    ap.add_argument("--crop", action="store_true", help="evaluate with garment isolation (crop=true)")
    ap.add_argument("--multicrop", choices=["off", "max_score", "mean_embedding"], default="off")
    ap.add_argument("--weights", default=None, help="override ranking weights 'visual,colour,pattern,attribute'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--real-photos", action="store_true",
                    help="mark the query set as independent real photos (not derived from catalogue images)")
    ap.add_argument("--update-docs", action="store_true", help="also write docs/EVALUATION.md for this run")
    args = ap.parse_args()

    with open(args.queries, newline="", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))
    queries = [q for q in queries if args.split == "all" or query_split(q) == args.split]
    if args.limit:
        queries = queries[: args.limit]
    with open(CATALOGUE_CSV, newline="", encoding="utf-8") as f:
        catalogue = {r["product_id"]: r for r in csv.DictReader(f)}
    s = get_settings()
    cands = args.candidates or [s.search_candidates]
    efs = args.ef_search or [None]
    configs = [(c, e) for c in cands for e in efs]
    w = None
    if args.weights:
        v = [float(x) for x in args.weights.split(",")]
        w = dict(zip(("visual", "colour", "pattern", "attribute"), v))
    eff_w = {k: round(x, 4) for k, x in (w and {k: x / sum(w.values()) for k, x in w.items()} or current_weights()).items()}
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    with connection() as conn:
        have = {r["model_key"]: r["n"] for r in vector_store.models_with_embeddings(conn)}
        models = args.models or [k for k, n in have.items() if n > 0]
        dsv = dataset_version(conn)
        pgv = ".".join(map(str, vector_store.pgvector_version(conn)))
        for key in models:
            if key not in MODEL_SPECS or not have.get(key):
                print(f"skip {key}: no catalogue embeddings (run generate_embeddings.py --model {key})")
                continue
            print(f"evaluating {key}: {len(queries)} queries x configs {configs}", flush=True)
            per = evaluate_model(conn, key, queries, catalogue, configs, args.crop, args.multicrop, w)
            stable = stability_check(conn, key, queries, cands[0])
            enc = get_registry().get(key)
            for (cand, ef), rows in per.items():
                stamp = datetime.now(timezone.utc)
                exp_id = f"{stamp:%Y%m%d-%H%M%S}_{key}_c{cand}" + (f"_ef{ef}" if ef else "") + f"_{args.tag}"
                meta = {
                    "id": exp_id, "generated": stamp.strftime("%Y-%m-%d %H:%M:%S UTC"), "tag": args.tag,
                    "model_key": key, "model_version": enc.version, "candidates": cand,
                    "ef_search": ef or vector_store.resolve_ef_search(cand), "ef_search_mode": "fixed" if ef else "auto",
                    "crop": args.crop, "multicrop": args.multicrop, "weights": eff_w, "split": args.split,
                    "dataset": dsv, "sources": sorted({r["source_dataset"] for r in catalogue.values()}),
                    "catalogue_csv_version": file_hash(CATALOGUE_CSV), "n_queries": len(rows),
                    "queries_file": os.path.relpath(args.queries, PROJECT_ROOT).replace("\\", "/"),
                    "query_set_version": file_hash(args.queries), "derived_queries": not args.real_photos,
                    "hardware": hardware(), "pgvector": pgv, "rss_mb": rss_mb(), "stable_order": stable,
                }
                res = {"meta": meta, **summarise(rows), "per_query": rows}
                d = EXPERIMENTS_DIR / exp_id
                d.mkdir(parents=True, exist_ok=False)
                (d / "result.json").write_text(json.dumps(res, indent=2, default=float))
                (d / "report.md").write_text(to_markdown(res))
                with open(EXPERIMENTS_DIR / "index.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": exp_id, **{k: meta[k] for k in ("generated", "model_key", "candidates",
                                        "ef_search", "crop", "multicrop", "split", "tag")},
                                        "hybrid": res["overall"]["hybrid"], "visual": res["overall"]["visual"],
                                        "p50": res["overall"]["latency_ms_p50"]}, default=float) + "\n")
                o = res["overall"]
                print(f"  [{exp_id}] visual R@1 {o['visual']['R@1']:.3f} | hybrid R@1 {o['hybrid']['R@1']:.3f} "
                      f"R@10 {o['hybrid']['R@10']:.3f} mAP {o['hybrid']['mAP']:.3f} | p50 {o['latency_ms_p50']:.0f} ms "
                      f"p95 {o['latency_ms_p95']:.0f} ms | stable={stable}")
                written.append(d)
                if args.update_docs:
                    (PROJECT_ROOT / "docs" / "EVALUATION.md").write_text(to_markdown(res))
    print(f"wrote {len(written)} experiment(s) to {EXPERIMENTS_DIR}. Compare with: python scripts/compare_experiments.py")


if __name__ == "__main__":
    main()
