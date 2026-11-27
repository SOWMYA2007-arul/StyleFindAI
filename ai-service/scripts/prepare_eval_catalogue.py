import csv
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.db import connection

AI_DIR = Path(__file__).resolve().parents[1]
ROOT = AI_DIR.parent
DATASET = ROOT / "dataset"
RAW = DATASET / "raw" / "generated-dresses"
METADATA = DATASET / "metadata"
PRODUCTS = DATASET / "products"

RAW_CSV = RAW / "images.csv"
OUTPUTS = [
    METADATA / "catalogue.csv",
    METADATA / "generated_dresses_catalogue.csv",
]

COLUMNS = [
    "product_id", "product_name", "category", "subcategory",
    "colour", "pattern", "sleeve_type", "neckline", "image_path",
    "source_dataset", "licence_information", "gender",
    "article_type", "usage", "attribute_source",
]

def main():
    for path in [RAW_CSV, METADATA, PRODUCTS]:
        if not path.exists():
            raise FileNotFoundError(f"Required path missing: {path}")

    for path in OUTPUTS:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    existing_images = [
        p for p in PRODUCTS.rglob("*")
        if p.is_file() and p.name != ".gitkeep"
    ]
    if existing_images:
        raise RuntimeError(
            f"Products folder is not empty: {existing_images[:3]}"
        )

    with RAW_CSV.open(newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    raw_by_id = {r["image"]: r for r in raw_rows}
    if len(raw_by_id) != len(raw_rows):
        raise RuntimeError("Duplicate image IDs in raw metadata CSV")

    image_files = {}
    for p in RAW.rglob("*.jpg"):
        if p.stem in image_files:
            raise RuntimeError(f"Duplicate raw image filename: {p.stem}")
        image_files[p.stem] = p

    query = """
        SELECT
            p.product_code,
            p.name,
            p.subcategory,
            p.source_dataset,
            p.licence_information,
            p.gender,
            c.slug AS category,
            pi.storage_key
        FROM products p
        JOIN categories c ON c.id = p.category_id
        JOIN product_images pi
          ON pi.product_id = p.id AND pi.is_primary
        WHERE p.product_code LIKE 'ai-dress-%'
          AND p.is_active
        ORDER BY p.product_code
    """

    with connection() as conn:
        db_rows = conn.execute(query).fetchall()

    if len(db_rows) != 215:
        raise RuntimeError(
            f"Expected 215 active products; found {len(db_rows)}"
        )

    catalogue_rows = []
    copy_plan = []

    for db in db_rows:
        code = db["product_code"]
        raw_id = code.replace("ai-", "", 1)
        raw = raw_by_id.get(raw_id)
        if raw is None:
            raise RuntimeError(f"No raw metadata for {code}")

        source = image_files.get(raw_id)
        if source is None:
            raise FileNotFoundError(f"No original JPG for {raw_id}")

        storage_key = Path(db["storage_key"])
        if storage_key.is_absolute() or ".." in storage_key.parts:
            raise RuntimeError(f"Unsafe storage key: {storage_key}")

        destination = PRODUCTS / storage_key
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite: {destination}")

        row = {
            "product_id": code,
            "product_name": db["name"],
            "category": db["category"],
            "subcategory": db["subcategory"] or "",
            "colour": raw.get("colour", "").strip().lower(),
            "pattern": raw.get("pattern", "").strip().lower(),
            "sleeve_type": raw.get("sleeve_type", "").strip().lower() or "unknown",
            "neckline": raw.get("neckline", "").strip().lower() or "unknown",
            "image_path": storage_key.as_posix(),
            "source_dataset": db["source_dataset"] or "",
            "licence_information": db["licence_information"] or "",
            "gender": db["gender"] or "",
            "article_type": raw.get("label", ""),
            "usage": "",
            "attribute_source": (
                "colour/pattern/design=generated prompt metadata (unverified)"
            ),
        }

        catalogue_rows.append(row)
        copy_plan.append((source, destination))

    # All database mappings and source files have been validated.
    copied = []
    try:
        for source, destination in copy_plan:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination)

        for output in OUTPUTS:
            with output.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(catalogue_rows)

    except Exception:
        for path in copied:
            if path.exists():
                path.unlink()
        for output in OUTPUTS:
            if output.exists():
                output.unlink()
        raise

    print("Catalogue preparation successful.")
    print(f"Products mapped: {len(catalogue_rows)}")
    print(f"Images copied: {len(copied)}")
    print(f"Catalogue files created: {len(OUTPUTS)}")
    for output in OUTPUTS:
        print(output)

if __name__ == "__main__":
    main()
