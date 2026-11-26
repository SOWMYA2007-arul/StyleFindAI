"""Print and save catalogue statistics (counts per category / colour / pattern, image sizes).

    python scripts/dataset_statistics.py
Writes dataset/metadata/statistics.json and docs/DATASET_STATISTICS.md
"""
from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np
from PIL import Image

from _common import CATALOGUE_CSV, PRODUCTS_DIR, PROJECT_ROOT


def main() -> None:
    with open(CATALOGUE_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sizes = []
    for r in rows:
        try:
            with Image.open(PRODUCTS_DIR / r["image_path"]) as im:
                sizes.append(im.size)
        except Exception:  # noqa: BLE001
            pass
    w = np.array([s[0] for s in sizes] or [0]); h = np.array([s[1] for s in sizes] or [0])
    stats = {
        "total_products": len(rows),
        "sources": dict(Counter(r["source_dataset"] for r in rows)),
        "by_category": dict(Counter(r["category"] for r in rows).most_common()),
        "by_subcategory": dict(Counter(r["subcategory"] for r in rows).most_common()),
        "by_colour": dict(Counter(r["colour"] for r in rows).most_common(20)),
        "by_pattern": dict(Counter(r["pattern"] for r in rows).most_common()),
        "by_sleeve_type": dict(Counter(r["sleeve_type"] for r in rows).most_common()),
        "by_neckline": dict(Counter(r["neckline"] for r in rows).most_common()),
        "unknown_rate": {a: round(sum(r[a] == "unknown" for r in rows) / max(1, len(rows)), 3)
                         for a in ("colour", "pattern", "sleeve_type", "neckline")},
        "image_width": {"min": int(w.min()), "median": int(np.median(w)), "max": int(w.max())},
        "image_height": {"min": int(h.min()), "median": int(np.median(h)), "max": int(h.max())},
    }
    (CATALOGUE_CSV.parent / "statistics.json").write_text(json.dumps(stats, indent=2))
    md = ["# Dataset statistics", "", f"Generated from `{CATALOGUE_CSV.relative_to(PROJECT_ROOT)}`.", "",
          f"**Total products:** {stats['total_products']}", "", f"**Sources:** {stats['sources']}", ""]
    for key in ("by_category", "by_subcategory", "by_colour", "by_pattern", "by_sleeve_type", "by_neckline"):
        md += [f"## {key.replace('_', ' ').title()}", "", "| value | count |", "|---|---|"]
        md += [f"| {k} | {v} |" for k, v in stats[key].items()] + [""]
    md += ["## Unknown-value rate", "", "| attribute | share unknown |", "|---|---|"]
    md += [f"| {k} | {v:.1%} |" for k, v in stats["unknown_rate"].items()]
    md += ["", f"Image width min/median/max: {stats['image_width']}  ", f"Image height: {stats['image_height']}"]
    (PROJECT_ROOT / "docs" / "DATASET_STATISTICS.md").write_text("\n".join(md) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
