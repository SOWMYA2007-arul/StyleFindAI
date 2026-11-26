-- Rollback of 003. Removes only what 003 added; products, images and embeddings are untouched.
ALTER TABLE products DROP CONSTRAINT IF EXISTS chk_products_name_not_blank;
DROP TRIGGER IF EXISTS trg_image_embeddings_indexed ON image_embeddings;
DROP FUNCTION IF EXISTS mark_image_indexed();
DROP TRIGGER IF EXISTS trg_product_images_invalidate ON product_images;
DROP FUNCTION IF EXISTS invalidate_image_derivatives();
DROP INDEX IF EXISTS idx_product_images_index_status;
ALTER TABLE product_images DROP CONSTRAINT IF EXISTS chk_index_status;
ALTER TABLE product_images DROP COLUMN IF EXISTS indexed_at;
ALTER TABLE product_images DROP COLUMN IF EXISTS index_error;
ALTER TABLE product_images DROP COLUMN IF EXISTS index_status;
ALTER TABLE admin_users DROP COLUMN IF EXISTS token_version;
-- embedding_models row 'cv-baseline'/'cv-v2' is kept if embeddings reference it
DELETE FROM embedding_models m WHERE m.model_key = 'cv-baseline' AND m.model_version = 'cv-v2'
  AND NOT EXISTS (SELECT 1 FROM image_embeddings e WHERE e.model_key = m.model_key AND e.model_version = m.model_version);
