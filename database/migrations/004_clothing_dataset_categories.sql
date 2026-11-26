-- 004_clothing_dataset_categories.sql
-- Adds the garment categories used by the public-domain Clothing Dataset import
-- (scripts/import_clothing_dataset.py). Only inserts rows; nothing existing is changed.
-- Rollback: python scripts/migrate.py --rollback 004_clothing_dataset_categories.sql
INSERT INTO categories (slug, name) VALUES
    ('outerwear', 'Outerwear & Jackets'),
    ('pants', 'Pants & Trousers'),
    ('shorts', 'Shorts'),
    ('skirts', 'Skirts'),
    ('footwear', 'Footwear'),
    ('hats', 'Hats & Caps')
ON CONFLICT (slug) DO NOTHING;
