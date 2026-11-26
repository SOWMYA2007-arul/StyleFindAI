"""Choose which products the website and search show, WITHOUT deleting anything (uses products.is_active).

    python scripts/catalogue_visibility.py --status
    python scripts/catalogue_visibility.py --only-source "clothing-dataset-full (Alexey Grigorev)"
    python scripts/catalogue_visibility.py --restore      # back to exactly how it was before the last change

Every change first saves the current active/inactive state of every product to
dataset/metadata/visibility_backup_<time>.json; --restore uses the newest backup.
Embeddings, images and attributes are never touched, so switching back is instant (no re-embedding).
"""
from __future__ import annotations

import argparse
import json
import time

from _common import DATASET_DIR
from app.services.db import connection

BACKUP_DIR = DATASET_DIR / "metadata"


def status(conn) -> None:
    rows = conn.execute("""SELECT source_dataset, count(*) AS total, count(*) FILTER (WHERE is_active) AS active
                           FROM products GROUP BY 1 ORDER BY 1""").fetchall()
    print(f"{'source_dataset':<55} {'active':>7} {'total':>7}")
    for r in rows:
        print(f"{r['source_dataset']:<55} {r['active']:>7} {r['total']:>7}")


def backup(conn) -> str:
    state = {r["product_code"]: r["is_active"] for r in conn.execute("SELECT product_code, is_active FROM products")}
    path = BACKUP_DIR / f"visibility_backup_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--status", action="store_true")
    g.add_argument("--only-source", metavar="SOURCE", help="show only products whose source_dataset equals SOURCE")
    g.add_argument("--all", action="store_true", help="show every product")
    g.add_argument("--restore", action="store_true", help="undo the last change")
    args = ap.parse_args()
    with connection() as conn:
        if args.status:
            return status(conn)
        if args.restore:
            files = sorted(BACKUP_DIR.glob("visibility_backup_*.json"))
            if not files:
                raise SystemExit("no backup found")
            state = json.loads(files[-1].read_text())
            for code, active in state.items():
                conn.execute("UPDATE products SET is_active = %s WHERE product_code = %s", (active, code))
            conn.commit()
            files[-1].rename(files[-1].with_suffix(".restored.json"))
            print(f"restored from {files[-1].name}")
            return status(conn)
        if args.only_source:
            n = conn.execute("SELECT count(*) AS n FROM products WHERE source_dataset = %s", (args.only_source,)).fetchone()["n"]
            if n == 0:
                status(conn)
                raise SystemExit(f"no products with source_dataset = '{args.only_source}' (names above)")
        saved = backup(conn)
        if args.all:
            conn.execute("UPDATE products SET is_active = TRUE")
        else:
            conn.execute("UPDATE products SET is_active = (source_dataset = %s)", (args.only_source,))
        conn.commit()
        print(f"previous state saved to {saved} (undo: --restore)")
        status(conn)


if __name__ == "__main__":
    main()
