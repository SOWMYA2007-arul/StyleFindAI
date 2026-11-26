"""Shared helpers for the dataset / embedding scripts."""
from __future__ import annotations

import sys
from pathlib import Path

AI_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AI_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

DATASET_DIR = PROJECT_ROOT / "dataset"
CATALOGUE_CSV = DATASET_DIR / "metadata" / "catalogue.csv"
PRODUCTS_DIR = DATASET_DIR / "products"

CATALOGUE_COLUMNS = [
    "product_id", "product_name", "category", "subcategory", "colour", "pattern",
    "sleeve_type", "neckline", "image_path", "source_dataset", "licence_information",
    # extra (useful, still real metadata):
    "gender", "article_type", "usage", "attribute_source",
]

# Our six catalogue groups -> dataset articleType values that belong in them.
CATEGORY_GROUPS: dict[str, list[str]] = {
    "kurtas-kurtis": ["Kurtas", "Kurtis", "Kurta Sets", "Tunics"],
    "sarees": ["Sarees"],
    "lehengas": ["Lehenga Choli"],
    "western-dresses": ["Dresses", "Jumpsuit"],
    "tops-shirts": ["Tops", "Shirts", "Tshirts"],
    "other-ethnic": ["Salwar", "Churidar", "Salwar and Dupatta", "Dupatta", "Patiala", "Shrug", "Nehru Jackets"],
}
DEFAULT_QUOTAS = {"kurtas-kurtis": 300, "sarees": 200, "lehengas": 150, "western-dresses": 200,
                  "tops-shirts": 150, "other-ethnic": 200}

# Explicit exclusions (defence in depth: anything not mapped above is dropped anyway).
EXCLUDED_MASTER = {"Footwear", "Accessories", "Personal Care", "Free Items", "Sporting Goods", "Home"}
EXCLUDED_SUB = {"Innerwear", "Loungewear and Nightwear", "Socks", "Bags", "Wallets", "Watches",
                "Jewellery", "Shoes", "Sandal", "Flip Flops", "Fragrance", "Makeup", "Lips", "Nails",
                "Skin", "Skin Care", "Hair", "Eyewear", "Belts", "Ties"}
EXCLUDED_ARTICLE = {"Bra", "Briefs", "Nightdress", "Night suits", "Lounge Pants", "Lounge Shorts",
                    "Robe", "Baby Dolls", "Camisoles", "Shapewear", "Innerwear Vests", "Bath Robe",
                    "Lounge Tshirts", "Boxers", "Trunk", "Lingerie Set"}


def slug_to_name(slug: str) -> str:
    return {"kurtas-kurtis": "Kurtas & Kurtis", "sarees": "Sarees", "lehengas": "Lehengas",
            "western-dresses": "Western Dresses", "tops-shirts": "Tops & Shirts",
            "other-ethnic": "Other Ethnic Wear"}.get(slug, slug)
