-- 003_integrity_and_index_status.sql
-- Additive and data-preserving: no rows are deleted, existing embeddings stay valid.
-- Rollback: python scripts/migrate.py --rollback 003_integrity_and_index_status.sql

-- 1) Indexing status per image (shown in the admin panel, incl. failures)
ALTER TABLE product_images ADD COLUMN IF NOT EXISTS index_status TEXT NOT NULL DEFAULT 'pending';
ALTER TABLE product_images ADD COLUMN IF NOT EXISTS index_error TEXT;
ALTER TABLE product_images ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMPTZ;
DO $$ BEGIN
    ALTER TABLE product_images ADD CONSTRAINT chk_index_status CHECK (index_status IN ('pending', 'indexed', 'failed'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- backfill from the embeddings that already exist
UPDATE product_images pi SET index_status = 'indexed',
       indexed_at = (SELECT max(e.created_at) FROM image_embeddings e WHERE e.product_image_id = pi.id)
WHERE EXISTS (SELECT 1 FROM image_embeddings e WHERE e.product_image_id = pi.id) AND pi.index_status <> 'indexed';
CREATE INDEX IF NOT EXISTS idx_product_images_index_status ON product_images(index_status) WHERE index_status <> 'indexed';

-- 2) An embedding can never describe a different image than the one stored:
--    when an image file changes (storage_key or sha256), its embeddings and colour/pattern features
--    are removed automatically in the SAME transaction, and it is marked for re-indexing.
CREATE OR REPLACE FUNCTION invalidate_image_derivatives() RETURNS trigger AS $$
BEGIN
    IF NEW.storage_key IS DISTINCT FROM OLD.storage_key OR NEW.sha256 IS DISTINCT FROM OLD.sha256 THEN
        DELETE FROM image_embeddings WHERE product_image_id = OLD.id;
        DELETE FROM image_features WHERE product_image_id = OLD.id;
        NEW.phash := CASE WHEN NEW.phash IS DISTINCT FROM OLD.phash THEN NEW.phash ELSE NULL END;
        NEW.index_status := 'pending';
        NEW.index_error := NULL;
        NEW.indexed_at := NULL;
    END IF;
    RETURN NEW;
END; $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_product_images_invalidate ON product_images;
CREATE TRIGGER trg_product_images_invalidate BEFORE UPDATE OF storage_key, sha256 ON product_images
    FOR EACH ROW EXECUTE FUNCTION invalidate_image_derivatives();

-- 3) Writing an embedding marks the image as indexed
CREATE OR REPLACE FUNCTION mark_image_indexed() RETURNS trigger AS $$
BEGIN
    UPDATE product_images SET index_status = 'indexed', index_error = NULL, indexed_at = now()
    WHERE id = NEW.product_image_id;
    RETURN NEW;
END; $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_image_embeddings_indexed ON image_embeddings;
CREATE TRIGGER trg_image_embeddings_indexed AFTER INSERT OR UPDATE OF embedding ON image_embeddings
    FOR EACH ROW EXECUTE FUNCTION mark_image_indexed();

-- 4) Admin session revocation: tokens carry token_version; logout / password reset increments it
ALTER TABLE admin_users ADD COLUMN IF NOT EXISTS token_version INTEGER NOT NULL DEFAULT 0;

-- 5) Register the v2 classical baseline (its features changed; v1 vectors are kept, never mixed)
INSERT INTO embedding_models (model_key, model_version, model_name, dims)
VALUES ('cv-baseline', 'cv-v2', 'Classical CV features v2 (improved garment mask)', 512)
ON CONFLICT DO NOTHING;

-- 6) Products must have a non-empty name (NOT VALID: existing rows are not re-checked, new ones are)
DO $$ BEGIN
    ALTER TABLE products ADD CONSTRAINT chk_products_name_not_blank CHECK (length(btrim(name)) > 0) NOT VALID;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
