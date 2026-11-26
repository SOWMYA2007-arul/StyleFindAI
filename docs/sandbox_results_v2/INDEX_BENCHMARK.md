# pgvector HNSW benchmark

Generated 2026-11-22 11:55 UTC by `scripts/benchmark_vector_index.py`. Queries = stored catalogue embeddings (self-match excluded from nothing; exact search is ground truth).

pgvector 0.6.0. `planner` = PostgreSQL's own choice; `forced` = seq/bitmap scans disabled.

## cv-baseline (cv-v2, 512-d, 360 vectors)

| filter | k | ef_search | planner plan | planner recall | planner p50 ms | forced recall | forced rows (min) | forced p50 ms | exact p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| none | 50 | 40 | Seq Scan on image_embeddings | 1.000 | 1.9 | 0.399 | 19 | 0.8 | 1.7 |
| none | 50 | 100 | Seq Scan on image_embeddings | 1.000 | 1.8 | 0.995 | 48 | 1.3 | 1.7 |
| none | 50 | 400 | Seq Scan on image_embeddings | 1.000 | 1.8 | 1.000 | 50 | 2.0 | 1.7 |
| none | 100 | 40 | Seq Scan on image_embeddings | 1.000 | 1.8 | 1.000 | 100 | 1.8 | 1.8 |
| none | 100 | 100 | Seq Scan on image_embeddings | 1.000 | 1.8 | 1.000 | 100 | 1.8 | 1.8 |
| none | 100 | 400 | Seq Scan on image_embeddings | 1.000 | 1.8 | 1.000 | 100 | 1.8 | 1.8 |
| none | 200 | 40 | Seq Scan on image_embeddings | 1.000 | 1.9 | 1.000 | 200 | 1.8 | 1.8 |
| none | 200 | 100 | Seq Scan on image_embeddings | 1.000 | 1.9 | 1.000 | 200 | 1.8 | 1.8 |
| none | 200 | 400 | Seq Scan on image_embeddings | 1.000 | 1.9 | 1.000 | 200 | 1.9 | 1.8 |
| category=western-dresses | 50 | 40 | Index Scan on image_embeddings_product_image_id_model_key_model_version_key | 1.000 | 0.7 | 1.000 | 40 | 0.7 | 0.8 |
| category=western-dresses | 50 | 100 | Index Scan on image_embeddings_product_image_id_model_key_model_version_key | 1.000 | 0.7 | 1.000 | 40 | 0.7 | 0.8 |
| category=western-dresses | 50 | 400 | Index Scan on image_embeddings_product_image_id_model_key_model_version_key | 1.000 | 0.7 | 1.000 | 40 | 0.7 | 0.8 |
| category=western-dresses | 100 | 40 | Index Scan on image_embeddings_product_image_id_model_key_model_version_key | 1.000 | 0.7 | 1.000 | 40 | 0.7 | 0.8 |
| category=western-dresses | 100 | 100 | Index Scan on image_embeddings_product_image_id_model_key_model_version_key | 1.000 | 0.7 | 1.000 | 40 | 0.7 | 0.8 |
| category=western-dresses | 100 | 400 | Index Scan on image_embeddings_product_image_id_model_key_model_version_key | 1.000 | 0.7 | 1.000 | 40 | 0.7 | 0.8 |
| category=western-dresses | 200 | 40 | Index Scan on image_embeddings_product_image_id_model_key_model_version_key | 1.000 | 0.7 | 1.000 | 40 | 0.7 | 0.8 |
| category=western-dresses | 200 | 100 | Index Scan on image_embeddings_product_image_id_model_key_model_version_key | 1.000 | 0.7 | 1.000 | 40 | 0.7 | 0.8 |
| category=western-dresses | 200 | 400 | Index Scan on image_embeddings_product_image_id_model_key_model_version_key | 1.000 | 0.7 | 1.000 | 40 | 0.7 | 0.8 |

## Reading the table

* If `planner plan` is a Seq Scan, PostgreSQL decided an exact scan is cheaper (normal for a few thousand
  vectors). Results are then exact; do not force the index.
* `forced rows (min)` below k with a filter shows the classic approximate-search + filter problem. The app
  then uses pgvector >= 0.8 iterative scans or falls back to an exact scan (FILTERED_EXACT_FALLBACK).
* Choose the smallest ef_search whose forced recall is ~1.0 for your k; keep HNSW_EF_SEARCH=0 (auto) otherwise.
