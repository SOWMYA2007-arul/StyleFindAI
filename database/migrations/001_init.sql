-- 001_init.sql : core StyleFind AI schema
-- Images are stored on disk (dataset/products); the DB keeps only storage keys.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ---------- catalogue ----------
CREATE TABLE IF NOT EXISTS categories (
    id          SERIAL PRIMARY KEY,
    slug        TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    parent_id   INTEGER REFERENCES categories(id) ON DELETE SET NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS products (
    id                  BIGSERIAL PRIMARY KEY,
    product_code        TEXT NOT NULL UNIQUE,         -- dataset product_id, e.g. "fpi-15970"
    name                TEXT NOT NULL,
    category_id         INTEGER NOT NULL REFERENCES categories(id) ON DELETE RESTRICT,
    subcategory         TEXT,
    gender              TEXT,
    price               NUMERIC(10,2) CHECK (price IS NULL OR price >= 0),
    description         TEXT,
    source_dataset      TEXT NOT NULL,
    licence_information TEXT NOT NULL,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id);
CREATE INDEX IF NOT EXISTS idx_products_active ON products(is_active);

CREATE TABLE IF NOT EXISTS product_images (
    id          BIGSERIAL PRIMARY KEY,
    product_id  BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    storage_key TEXT NOT NULL UNIQUE,                 -- path relative to PRODUCTS_DIR, never absolute
    is_primary  BOOLEAN NOT NULL DEFAULT TRUE,
    width       INTEGER CHECK (width > 0),
    height      INTEGER CHECK (height > 0),
    sha256      TEXT NOT NULL,
    phash       TEXT,                                 -- 64-bit perceptual hash, hex
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_product_images_product ON product_images(product_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_product_images_sha ON product_images(sha256);
-- one primary image per product
CREATE UNIQUE INDEX IF NOT EXISTS uq_product_primary_image ON product_images(product_id) WHERE is_primary;

-- Verified catalogue attributes (metadata / admin) and model predictions are kept apart via `source` + `verified`.
CREATE TABLE IF NOT EXISTS product_attributes (
    product_id  BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    attribute   TEXT NOT NULL,       -- colour, pattern, sleeve_type, neckline, style, fabric, embroidery ...
    value       TEXT NOT NULL,       -- 'unknown' allowed
    source      TEXT NOT NULL CHECK (source IN ('catalogue_metadata','name_keywords','admin','model_prediction')),
    confidence  REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    verified    BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (product_id, attribute, source)
);
CREATE INDEX IF NOT EXISTS idx_attr_lookup ON product_attributes(attribute, value);

-- ---------- embeddings ----------
CREATE TABLE IF NOT EXISTS embedding_models (
    model_key     TEXT NOT NULL,     -- fashion-clip | openclip-b32 | dinov2-b14 | cv-baseline
    model_version TEXT NOT NULL,     -- checkpoint / revision id
    model_name    TEXT NOT NULL,     -- human readable / HF repo id
    dims          INTEGER NOT NULL CHECK (dims > 0),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_key, model_version)
);

-- `embedding` has no fixed dimension so several models can coexist.
-- Per-model HNSW indexes cast to the correct dimension (see 002_vector_indexes.sql).
CREATE TABLE IF NOT EXISTS image_embeddings (
    id               BIGSERIAL PRIMARY KEY,
    product_image_id BIGINT NOT NULL REFERENCES product_images(id) ON DELETE CASCADE,
    model_key        TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    dims             INTEGER NOT NULL,
    embedding        vector NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (model_key, model_version) REFERENCES embedding_models(model_key, model_version) ON DELETE CASCADE,
    UNIQUE (product_image_id, model_key, model_version),
    CHECK (vector_dims(embedding) = dims)
);
CREATE INDEX IF NOT EXISTS idx_emb_model ON image_embeddings(model_key, model_version);

-- Model-independent colour / texture descriptors used by the hybrid re-ranker.
CREATE TABLE IF NOT EXISTS image_features (
    product_image_id BIGINT PRIMARY KEY REFERENCES product_images(id) ON DELETE CASCADE,
    feature_version  TEXT NOT NULL,
    colour_palette   JSONB NOT NULL,     -- [{"lab":[L,a,b],"weight":w,"name":"blue"}, ...]
    colour_name      TEXT,               -- dominant colour name derived from pixels
    pattern_vector   REAL[] NOT NULL,    -- texture descriptor (LBP + edge statistics)
    pattern_label    TEXT,               -- solid / textured (heuristic)
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------- search logging (anonymous, optional) ----------
CREATE TABLE IF NOT EXISTS search_history (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_key            TEXT NOT NULL,
    query_image_sha256   TEXT,           -- hash only; the uploaded image is NOT stored
    text_query           TEXT,
    predicted_attributes JSONB,          -- predicted QUERY attributes (never mixed with catalogue)
    filters              JSONB,
    latency_ms           INTEGER,
    result_count         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_search_history_created ON search_history(created_at);

CREATE TABLE IF NOT EXISTS search_results (
    search_id    UUID NOT NULL REFERENCES search_history(id) ON DELETE CASCADE,
    rank         INTEGER NOT NULL CHECK (rank > 0),
    product_id   BIGINT REFERENCES products(id) ON DELETE SET NULL,
    final_score  REAL NOT NULL,
    visual_score REAL NOT NULL,
    match_type   TEXT NOT NULL CHECK (match_type IN ('exact','similar','related')),
    PRIMARY KEY (search_id, rank)
);

-- ---------- admin ----------
CREATE TABLE IF NOT EXISTS admin_users (
    id            SERIAL PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,          -- bcrypt
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at TIMESTAMPTZ
);

-- keep updated_at fresh
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END; $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_products_touch ON products;
CREATE TRIGGER trg_products_touch BEFORE UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
DROP TRIGGER IF EXISTS trg_product_images_touch ON product_images;
CREATE TRIGGER trg_product_images_touch BEFORE UPDATE ON product_images
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

-- seed the six catalogue groups used by the dataset scripts
INSERT INTO categories (slug, name) VALUES
    ('kurtas-kurtis', 'Kurtas & Kurtis'),
    ('sarees', 'Sarees'),
    ('lehengas', 'Lehengas'),
    ('western-dresses', 'Western Dresses'),
    ('tops-shirts', 'Tops & Shirts'),
    ('other-ethnic', 'Other Ethnic Wear')
ON CONFLICT (slug) DO NOTHING;
