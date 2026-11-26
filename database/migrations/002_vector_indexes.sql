-- 002_vector_indexes.sql : per-model HNSW indexes (cosine distance)
-- The `embedding` column has no fixed size, so each index casts to its model's dimension
-- and is restricted (partial index) to rows of that model. Queries must use the SAME
-- cast and WHERE model_key = '<key>' for PostgreSQL to use the index.

INSERT INTO embedding_models (model_key, model_version, model_name, dims) VALUES
    ('fashion-clip', 'hf:patrickjohncyh/fashion-clip', 'FashionCLIP (CLIP ViT-B/32 fine-tuned on fashion)', 512),
    ('openclip-b32', 'openclip:ViT-B-32/laion2b_s34b_b79k', 'OpenCLIP ViT-B/32 (LAION-2B)', 512),
    ('dinov2-b14', 'hf:facebook/dinov2-base', 'DINOv2 ViT-B/14 (CLS token)', 768),
    ('cv-baseline', 'cv-v1', 'Classical CV features (HSV hist + LBP + HOG)', 512)
ON CONFLICT DO NOTHING;

CREATE INDEX IF NOT EXISTS hnsw_fashion_clip ON image_embeddings
    USING hnsw ((embedding::vector(512)) vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE model_key = 'fashion-clip';

CREATE INDEX IF NOT EXISTS hnsw_openclip_b32 ON image_embeddings
    USING hnsw ((embedding::vector(512)) vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE model_key = 'openclip-b32';

CREATE INDEX IF NOT EXISTS hnsw_dinov2_b14 ON image_embeddings
    USING hnsw ((embedding::vector(768)) vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE model_key = 'dinov2-b14';

CREATE INDEX IF NOT EXISTS hnsw_cv_baseline ON image_embeddings
    USING hnsw ((embedding::vector(512)) vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE model_key = 'cv-baseline';
