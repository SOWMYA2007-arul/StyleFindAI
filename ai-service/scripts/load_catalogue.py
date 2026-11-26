"""Load dataset/metadata/catalogue.csv into PostgreSQL (products, product_images, product_attributes).

Idempotent: re-running updates existing rows (matched by product_code / storage_key) instead of
creating duplicates.

    python scripts/load_catalogue.py
"""
from __future__ import annotations

import csv

from PIL import Image

from _common import CATALOGUE_CSV, PRODUCTS_DIR
from app.services.db import connection
from remove_duplicates import file_sha256, image_phash


def main() -> None:
    with open(CATALOGUE_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    inserted = updated = 0
    with connection() as conn:
        cats = {r["slug"]: r["id"] for r in conn.execute("SELECT id, slug FROM categories")}
        for r in rows:
            path = PRODUCTS_DIR / r["image_path"]
            with Image.open(path) as im:
                w, h = im.size
            res = conn.execute(
                """INSERT INTO products (product_code, name, category_id, subcategory, gender, source_dataset,
                                         licence_information)
                   VALUES (%(code)s, %(name)s, %(cat)s, %(sub)s, %(gender)s, %(src)s, %(lic)s)
                   ON CONFLICT (product_code) DO UPDATE SET name = EXCLUDED.name, category_id = EXCLUDED.category_id,
                       subcategory = EXCLUDED.subcategory, gender = EXCLUDED.gender,
                       source_dataset = EXCLUDED.source_dataset, licence_information = EXCLUDED.licence_information
                   RETURNING id, (xmax = 0) AS inserted""",
                {"code": r["product_id"], "name": r["product_name"], "cat": cats[r["category"]],
                 "sub": r["subcategory"], "gender": r.get("gender"), "src": r["source_dataset"],
                 "lic": r["licence_information"]}).fetchone()
            pid = res["id"]
            inserted += res["inserted"]; updated += not res["inserted"]
            conn.execute(
                """INSERT INTO product_images (product_id, storage_key, is_primary, width, height, sha256, phash)
                   VALUES (%s, %s, TRUE, %s, %s, %s, %s)
                   ON CONFLICT (storage_key) DO UPDATE SET width = EXCLUDED.width, height = EXCLUDED.height,
                       sha256 = EXCLUDED.sha256, phash = EXCLUDED.phash""",
                (pid, r["image_path"], w, h, file_sha256(path), str(image_phash(path))))
            src_for = {"colour": "catalogue_metadata"}
            for attr in ("colour", "pattern", "sleeve_type", "neckline"):
                src = src_for.get(attr, "name_keywords")
                conn.execute(
                    """INSERT INTO product_attributes (product_id, attribute, value, source, verified)
                       VALUES (%s, %s, %s, %s, TRUE)
                       ON CONFLICT (product_id, attribute, source) DO UPDATE SET value = EXCLUDED.value""",
                    (pid, attr, (r.get(attr) or "unknown").strip().lower() or "unknown", src))
        conn.commit()
        total = conn.execute("SELECT count(*) AS n FROM products").fetchone()["n"]
    print(f"inserted {inserted}, updated {updated}; products in DB: {total}")


if __name__ == "__main__":
    main()
