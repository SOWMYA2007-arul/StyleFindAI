"""Compare stored evaluation experiments -> docs/EXPERIMENTS.md + experiments/comparison.json

    python scripts/compare_experiments.py                       # all experiments
    python scripts/compare_experiments.py --model fashion-clip  # one model
    python scripts/compare_experiments.py --tag candidates      # one experiment family

Keep your previous evaluation as the baseline (old single-file format, e.g. your FashionCLIP run):
    python scripts/compare_experiments.py --import-legacy ..\\backups\\results_fashionclip_baseline.json --label fashionclip-baseline-v1

Only experiments evaluated on the SAME query-set version and dataset version are directly
comparable; the report groups them and warns otherwise.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from _common import DATASET_DIR, PROJECT_ROOT

EXPERIMENTS_DIR = DATASET_DIR / "evaluation" / "experiments"


def load_all() -> list[dict]:
    out = []
    for f in sorted(EXPERIMENTS_DIR.glob("*/result.json")):
        try:
            r = json.loads(f.read_text())
            if "meta" not in r or "overall" not in r:
                continue  # e.g. weight-tuning results (different format, see docs/WEIGHT_TUNING.md)
            r.pop("per_query", None)
            out.append(r)
        except json.JSONDecodeError:
            print(f"warning: unreadable {f}")
    return out


def import_legacy(path: Path, label: str) -> Path:
    """Convert the v1 results.json (meta + results[]) into experiment folders without changing numbers."""
    data = json.loads(path.read_text())
    meta, created = data.get("meta", {}), None
    for r in data.get("results", []):
        exp_id = f"legacy_{r['model_key']}_{label}"
        d = EXPERIMENTS_DIR / exp_id
        if d.exists():
            print(f"already imported: {exp_id}")
            continue
        d.mkdir(parents=True)
        res = {
            "meta": {"id": exp_id, "generated": meta.get("generated", "unknown"), "tag": label, "legacy": True,
                     "model_key": r["model_key"], "model_version": r.get("model_version", "unknown"),
                     "candidates": "50 (visual) / max(50, SEARCH_CANDIDATES) (hybrid)", "ef_search": "max(100, 2k)",
                     "crop": False, "multicrop": "off", "weights": "settings at the time", "split": "all",
                     "dataset": {"active_products": meta.get("catalogue_size"), "hash": "unknown (legacy)"},
                     "sources": meta.get("sources", []), "n_queries": meta.get("n_queries"),
                     "queries_file": meta.get("queries_file"), "query_set_version": "unknown (legacy)",
                     "derived_queries": True, "hardware": meta.get("device"), "imported_from": str(path),
                     "imported_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
            "overall": r["overall"], "by_transform": r.get("by_transform", {}),
        }
        (d / "result.json").write_text(json.dumps(res, indent=2))
        created = d
        print(f"imported {exp_id}")
    return created


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--tag")
    ap.add_argument("--import-legacy", type=Path); ap.add_argument("--label", default="baseline")
    args = ap.parse_args()
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.import_legacy:
        import_legacy(args.import_legacy, args.label)
    exps = [e for e in load_all() if (not args.model or e["meta"]["model_key"] == args.model)
            and (not args.tag or e["meta"].get("tag") == args.tag or e["meta"].get("legacy"))]
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    groups = defaultdict(list)
    for e in exps:
        groups[(e["meta"].get("query_set_version"), e["meta"]["dataset"].get("hash"))].append(e)
    L = ["# Experiment comparison", "",
         f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `scripts/compare_experiments.py` from "
         f"{len(exps)} stored experiments. Numbers are copied from each experiment's result.json.", "",
         "Rows in the same table used the same query set and catalogue version and are directly comparable.", ""]
    best_overall = []
    for (qv, dv), es in groups.items():
        es.sort(key=lambda e: e["meta"]["generated"])
        legacy = any(e["meta"].get("legacy") for e in es)
        L += [f"## Query set `{qv}`, catalogue `{dv}`" + (" (legacy import)" if legacy else ""), "",
              "| experiment | model | N | ef | crop | multicrop | split | visual R@1 | hybrid R@1 | R@5 | R@10 | mAP | p50 ms | p95 ms |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for e in es:
            m, o = e["meta"], e["overall"]
            h, v = o["hybrid"], o["visual"]
            p50 = o.get("latency_ms_p50"); p95 = o.get("latency_ms_p95")
            L.append(f"| {m['id']} | {m['model_key']} | {m['candidates']} | {m.get('ef_search')} | {m.get('crop')} | "
                     f"{m.get('multicrop')} | {m.get('split')} | {pct(v['R@1'])} | {pct(h['R@1'])} | {pct(h['R@5'])} | "
                     f"{pct(h['R@10'])} | {h['mAP']:.3f} | {p50 and round(p50)} | {p95 and round(p95)} |")
        best = max(es, key=lambda e: (round(e["overall"]["hybrid"]["mAP"], 3), -(e["overall"].get("latency_ms_p50") or 1e9)))
        L += ["", f"Best in this group by hybrid mAP (ties -> lower latency): **{best['meta']['id']}**", ""]
        best_overall.append(best["meta"]["id"])
    L += ["## How to choose", "",
          "* Prefer the setting with the best hybrid mAP / R@1 on the **val** split, then confirm it on **test**.",
          "* If two settings are equal within ~1 percentage point, choose the faster one (p95 latency).",
          "* All queries are augmented catalogue photos unless an experiment says otherwise; they are easier than",
          "  real customer photos."]
    (PROJECT_ROOT / "docs" / "EXPERIMENTS.md").write_text("\n".join(L) + "\n")
    (EXPERIMENTS_DIR / "comparison.json").write_text(json.dumps(
        {"generated": datetime.now(timezone.utc).isoformat(), "best_per_group": best_overall,
         "experiments": [{"meta": e["meta"], "overall": e["overall"]} for e in exps]}, indent=2, default=str))
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
