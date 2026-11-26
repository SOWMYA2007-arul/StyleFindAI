"""Recompute ONLY the colour/pattern features (image_features table) with the current algorithms.

Safe: products, images and embeddings are not touched; no model is loaded. Needed after upgrading,
because the improved foreground/colour/pattern code must describe catalogue images exactly like
query images.

    python scripts/recompute_features.py            # images whose features are missing or outdated
    python scripts/recompute_features.py --force    # all images
"""
from __future__ import annotations

import argparse
import time

from _common import PROJECT_ROOT  # noqa: F401  (sets sys.path)
from app.services.db import connection
from app.services.indexing import FEATURE_VERSION, recompute_features, stale_feature_images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    with connection() as conn:
        rows = stale_feature_images(conn, force=args.force)
        print(f"feature version {FEATURE_VERSION}: {len(rows)} images to update")
        t = time.perf_counter()
        n = recompute_features(conn, rows, progress=True)
    print(f"updated {n} images in {time.perf_counter() - t:.1f}s (embeddings unchanged)")


if __name__ == "__main__":
    main()
