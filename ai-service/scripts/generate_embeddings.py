"""Generate catalogue embeddings offline, in batches, and store them in pgvector.

    python scripts/generate_embeddings.py --model fashion-clip          # only missing ones
    python scripts/generate_embeddings.py --model fashion-clip --force  # regenerate all
    python scripts/generate_embeddings.py --model openclip-b32 --no-features

Colour / pattern features are model-independent; they are (re)computed with the first model
you run unless --no-features is given.

On Google Colab (GPU): set AI_DEVICE=cuda in .env; the same script runs unchanged.
"""
from __future__ import annotations

import argparse
import json

from _common import PROJECT_ROOT  # noqa: F401  (sets sys.path)
from app.models.encoders import MODEL_SPECS, get_registry
from app.services.db import connection
from app.services.indexing import index_images, pending_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help=f"one of {list(MODEL_SPECS)} (default: AI_PRIMARY_MODEL)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-features", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    from app.config import get_settings

    key = args.model or get_settings().ai_primary_model
    enc = get_registry().get(key)
    with connection() as conn:
        rows = pending_images(conn, enc, force=args.force)
        print(f"{key} ({enc.version}, {enc.dims}-d): {len(rows)} images to embed")
        res = index_images(conn, enc, rows, batch_size=args.batch_size, with_features=not args.no_features,
                           progress=True)
        n = conn.execute("SELECT count(*) AS n FROM image_embeddings WHERE model_key=%s AND model_version=%s",
                         (enc.key, enc.version)).fetchone()["n"]
    res["total_in_db_for_model"] = n
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
