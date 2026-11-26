-- Rollback of 002: drops the HNSW indexes only (queries still work, as exact scans). Model rows are kept.
DROP INDEX IF EXISTS hnsw_fashion_clip;
DROP INDEX IF EXISTS hnsw_openclip_b32;
DROP INDEX IF EXISTS hnsw_dinov2_b14;
DROP INDEX IF EXISTS hnsw_cv_baseline;
