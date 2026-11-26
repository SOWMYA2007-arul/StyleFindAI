"""Tune hybrid-ranking weights on a VALIDATION split and report them on a separate TEST split.

    python scripts/tune_weights.py --model fashion-clip
    python scripts/tune_weights.py --model fashion-clip --candidates 50 --step 0.05 --min-visual 0.4

How it avoids fooling itself:
  * queries are split by PRODUCT (all photos of one product are in the same split), see evaluate.py
  * component scores (visual / colour / pattern / attribute) are computed once per query, then every
    weight combination is scored offline - same candidates, same features
  * the best weights are chosen on val only; test numbers are reported for both the current and the
    best weights. If test does not improve, keep your current weights.
  * results are written to experiments/weights_<date>/ - .env is NEVER modified.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from datetime import datetime, timezone

import numpy as np

from _common import DATASET_DIR, PROJECT_ROOT
from app.config import get_settings
from app.services import vector_store
from app.services.db import connection
from app.services.preprocess import load_and_validate
from app.services.ranking import score_candidates, weights as current_weights
from app.services.search import build_context, prepare_query, retrieve
from evaluate import EXPERIMENTS_DIR, query_split

KEYS = ("visual", "colour", "pattern", "attribute")


def collect(conn, model: str, queries: list[dict], candidates: int):
    """Per query: matrix (n_candidates x 4) of components (NaN = missing) + index of the truth (or -1)."""
    out = []
    for i, q in enumerate(queries):
        loaded = load_and_validate((DATASET_DIR / "evaluation" / q["query_path"]).read_bytes(), 20 << 20)
        pq = prepare_query(loaded, model)
        cands = retrieve(conn, pq, candidates)
        details = vector_store.product_details(conn, [c["product_id"] for c in cands])
        scored = score_candidates(build_context(pq, details, cands), cands, details)
        M = np.array([[r["components"][k] if r["components"][k] is not None else np.nan for k in KEYS] for r in scored])
        ids = np.array([r["product"]["id"] for r in scored])
        cos = np.array([r["cosine"] for r in scored])
        codes = [r["product"]["product_code"] for r in scored]
        truth = codes.index(q["product_id"]) if q["product_id"] in codes else -1
        out.append({"M": M, "ids": ids, "cos": cos, "truth": truth, "split": query_split(q)})
        if (i + 1) % 50 == 0:
            print(f"  collected {i + 1}/{len(queries)}", flush=True)
    return out


def ranks_for(data, w: np.ndarray) -> list[int | None]:
    res = []
    for d in data:
        if d["truth"] < 0:
            res.append(None); continue
        M = d["M"]
        avail = ~np.isnan(M)
        num = np.nansum(np.where(avail, M, 0) * w, axis=1)
        den = (avail * w).sum(axis=1)
        final = np.where(den > 0, num / np.where(den > 0, den, 1), 0)
        # same deterministic order as ranking.sort_key: final desc, cosine desc, id asc
        order = np.lexsort((d["ids"], -np.round(d["cos"], 9), -np.round(final, 9)))
        res.append(int(np.where(order == d["truth"])[0][0]) + 1)
    return res


def metrics(r: list) -> dict:
    return {"R@1": float(np.mean([x is not None and x <= 1 for x in r])),
            "R@5": float(np.mean([x is not None and x <= 5 for x in r])),
            "R@10": float(np.mean([x is not None and x <= 10 for x in r])),
            "mAP": float(np.mean([1 / x if x else 0 for x in r])), "n": len(r)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--queries", default=str(DATASET_DIR / "evaluation" / "queries.csv"))
    ap.add_argument("--candidates", type=int, default=None)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--min-visual", type=float, default=0.4)
    args = ap.parse_args()
    s = get_settings()
    model = args.model or s.ai_primary_model
    cand = args.candidates or s.search_candidates
    with open(args.queries, newline="", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))
    with connection() as conn:
        data = collect(conn, model, queries, cand)
    val = [d for d in data if d["split"] == "val"]
    test = [d for d in data if d["split"] == "test"]
    cur = np.array([current_weights()[k] for k in KEYS])
    steps = int(round(1 / args.step))
    combos = []
    for v, c, p in itertools.product(range(steps + 1), repeat=3):
        a = steps - v - c - p
        if a >= 0 and v * args.step >= args.min_visual - 1e-9:
            combos.append(np.array([v, c, p, a]) * args.step)
    print(f"{len(val)} val / {len(test)} test queries; trying {len(combos)} weight combinations")
    scored = []
    for w in combos:
        m = metrics(ranks_for(val, w))
        scored.append((round(m["mAP"], 4), round(m["R@1"], 4), -float(np.abs(w - cur).sum()), w, m))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    best_w, best_val = scored[0][3], scored[0][4]
    res = {
        "generated": datetime.now(timezone.utc).isoformat(), "model": model, "candidates": cand,
        "queries_file": args.queries, "step": args.step, "min_visual": args.min_visual,
        "current_weights": dict(zip(KEYS, cur.round(4).tolist())), "best_weights": dict(zip(KEYS, best_w.round(4).tolist())),
        "val": {"current": metrics(ranks_for(val, cur)), "best": best_val},
        "test": {"current": metrics(ranks_for(test, cur)), "best": metrics(ranks_for(test, best_w))},
        "top10_on_val": [{"weights": dict(zip(KEYS, x[3].round(3).tolist())), "val_mAP": x[0], "val_R@1": x[1]} for x in scored[:10]],
    }
    gain = res["test"]["best"]["mAP"] - res["test"]["current"]["mAP"]
    res["recommendation"] = ("adopt best weights" if gain > 0.005 else
                             "keep current weights (no reliable improvement on the test split)")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    d = EXPERIMENTS_DIR / f"weights_{stamp}_{model}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(res, indent=2))
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    md = [f"# Ranking-weight tuning ({model}, N={cand})", "", f"Generated {res['generated']}. Chosen on val, reported on test.", "",
          "| weights | split | R@1 | R@5 | R@10 | mAP |", "|---|---|---|---|---|---|"]
    for name, w in (("current", res["current_weights"]), ("best on val", res["best_weights"])):
        key = "current" if name == "current" else "best"
        for sp in ("val", "test"):
            m = res[sp][key]
            md.append(f"| {name} {w} | {sp} | {pct(m['R@1'])} | {pct(m['R@5'])} | {pct(m['R@10'])} | {m['mAP']:.3f} |")
    md += ["", f"**Recommendation:** {res['recommendation']}.", ""]
    if res["recommendation"].startswith("adopt"):
        b = res["best_weights"]
        md += ["To adopt, edit your .env (not done automatically):", "```",
               f"RANK_W_VISUAL={b['visual']}", f"RANK_W_COLOUR={b['colour']}", f"RANK_W_PATTERN={b['pattern']}",
               f"RANK_W_ATTRIBUTE={b['attribute']}", "```"]
    (d / "report.md").write_text("\n".join(md) + "\n")
    (PROJECT_ROOT / "docs" / "WEIGHT_TUNING.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
