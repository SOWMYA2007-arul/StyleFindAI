# Build report: phases 1-13

This records what was built, what was run, and the **actual** results in the build environment.

## Build environment (and why it matters)

The project was built in a Linux cloud workspace, not on your Windows laptop:

* Available: Python 3.13, Node 22, PostgreSQL 16 + pgvector 0.6.0, PyTorch 2.15 CPU, Chrome (for UI tests).
* **Blocked by the workspace network:** Kaggle and Hugging Face. So the real product dataset and the real
  FashionCLIP / OpenCLIP / DINOv2 weights could not be downloaded there.

To still test every layer honestly:

* the dataset scripts, database, AI service, ranking, API, frontend and admin were run end-to-end on a
  **synthetic test fixture** (generated drawings of garments in the Kaggle file layout, every row tagged `SYNTHETIC_TEST_FIXTURE`);
* search was run with the **`cv-baseline` encoder** (real, weight-free OpenCV features) because neural weights were unavailable;
* the FashionCLIP class was run with a randomly initialised CLIP ViT-B/32 to check shapes, preprocessing and normalisation.

**Your remaining step:** on the laptop, follow README 1.4-1.6 with the Kaggle data and `fashion-clip`, then run
`evaluate.py` to produce real FashionCLIP vs OpenCLIP vs DINOv2 numbers. No numbers for those models are claimed anywhere.

---

## Phase 1: structure and environment
* **Built:** folder structure from the brief plus `scripts/` (PowerShell helpers); `.env.example`, `.gitignore` (excludes `.env`, weights, datasets, node_modules), `docker-compose.yml` (pgvector/pgvector:pg16).
* **Your machine:** could not be inspected from the build environment. `scripts/check-environment.ps1` reports Python, Node, npm, Docker, PostgreSQL, RAM, GPU and disk.
* **Compatibility findings:** see `docs/MODELS.md` (YOLO has no garment classes in COCO; Ultralytics is AGPL; Qwen2.5-VL-3B is non-commercial and too heavy for CPU-first use; pgvector on Windows needs Docker or a Visual Studio build).

## Phase 2: dataset
* **Files:** `ai-service/scripts/{_common,prepare_dataset,remove_duplicates,validate_images,dataset_statistics,make_test_fixtures}.py`, `app/utils/keywords.py`.
* **Result on the fixture (367 raw rows):** excluded exactly the planted junk: footwear 1, innerwear 1, duplicate id 1, men's 1, missing image 1, corrupt image 1, duplicate image 1. Selected 360. `validate_images.py` printed **PASSED**.
* **Bug found and fixed:** the first duplicate detector flagged 174 "duplicates" because pHash ignores colour (same cut, different colour). Now a pHash match must also pass a colour-thumbnail check. Result: exactly the 1 planted duplicate.

## Phase 3: PostgreSQL + pgvector
* **Files:** `database/migrations/001_init.sql`, `002_vector_indexes.sql`, `database/schema.sql`, `ai-service/scripts/migrate.py`, `load_catalogue.py`, `app/services/db.py`.
* Tables: categories, products, product_images, product_attributes (source + verified flag), embedding_models, image_embeddings (model_key + version + dims + CHECK), image_features, search_history, search_results, admin_users, schema_migrations.
* **Verified:** migrations apply once (second run reports "applied"); loading twice gives `inserted 360` then `updated 360` (no duplicates).
* **Bug found and fixed:** forcing the HNSW index showed it returned only 40 of 50 requested rows (pgvector `ef_search` default 40). The search now sets `hnsw.ef_search` per query.

## Phase 4: embeddings
* **Files:** `app/models/encoders.py` (FashionCLIP, OpenCLIP, DINOv2, cv-baseline; lazy LRU registry), `app/services/indexing.py`, `scripts/generate_embeddings.py`.
* **Result:** 360 images embedded in 13.4 s (cv-baseline + colour/pattern features); second run embedded 0 (only missing images are processed).

## Phase 5: first image-to-image search API (proved before the frontend)
* **Files:** `app/main.py`, `app/api/routes.py`, `app/services/{search,vector_store,preprocess}.py`.
* **Proof:** uploaded `syn-900007__background.jpg` (product 900007 pasted on a new background) with `curl -F image=@...`
  to `/ai/search`. Rank 1 was `syn-900007`, from real pgvector rows. Total 83 ms (embed 19 ms, vector search 5 ms).

## Phase 6: attributes and garment detection
* **Files:** `app/services/{colour,pattern,foreground,attributes,detection}.py`.
* Pixel colour palette (k-means in CIELAB), texture descriptor, zero-shot prompts for CLIP models (with confidence threshold → `unknown`), neighbour-vote category for text-less models, optional Qwen2.5-VL, optional YOLO → SAM 2 with full-image fallback.
* Tested: palette and pattern unit tests. YOLO / SAM 2 / Qwen were not executed (weights unavailable); they are off by default and every failure falls back safely.

## Phase 7: hybrid ranking
* **File:** `app/services/ranking.py`. Weights from `.env` (0.65/0.15/0.10/0.10), every component in [0,1], missing components dropped and weights renormalised.
* **Exact match rule:** exactly one of the 5 nearest items must be a near-identical photo (pHash ≤ 6 bits **and** colour check), or a very high cosine with a clear margin. Disabled when the user asks for a text modification.
* **Bugs found and fixed via evaluation:** exact-match precision was 85% (pHash colour-blindness) → 94.4% (colour check) → **100%** (uniqueness rule). Colour similarity was too strict (two reds scored 0.72) → Gaussian falloff.

## Phase 8: React frontend
* **Files:** `frontend/src/**`: 9 pages (home, visual search, catalogue, results, product details, about, admin login, admin dashboard, product management) plus a 404 page, components, contexts, hooks, typed API client.
* Every button is wired to a real endpoint; results show real stored images, match type, score breakdown and reasons.

## Phase 9: Node backend
* **Files:** `backend/src/**`: config, pg pool, validation (zod), multer memory uploads + magic-byte check, JWT admin auth, rate limits, helmet, CORS whitelist, error handler without leaks, AI client with timeouts, search history (hash only).

## Phase 10: admin dashboard
* Login, statistics, model/index table with Generate / Regenerate, product list, add / edit (with image replacement and re-indexing) / delete with inline confirmation, duplicate code and image protection.

## Phase 11: image + text search
* Text is parsed for colour / pattern / sleeve / neckline; CLIP models apply a text-delta edit to the query embedding; text-less models fall back to attribute re-ranking and say so in the response.
* Test: adding "similar design in blue colour" increased the number of blue products in the top 10 (`test_text_modification_changes_colour_ranking`).

## Phase 12: evaluation and testing (actual results)

| Suite | Command | Result |
|---|---|---|
| AI service unit + dataset + API + 8 scenarios | `python -m pytest tests` | **41 passed, 1 skipped** (skip = "≥ 1000 products" check, not applicable to the 360-item fixture) |
| Catalogue validation | `validate_images.py` | **PASSED** (360/360 images) |
| Backend typecheck | `npx tsc --noEmit` | clean |
| Backend integration (real DB + AI) | `npm test` | **13 passed** |
| Frontend production build | `npm run build` | built, 0 type errors |
| Browser end-to-end | `node tests/ui-smoke.mjs` | **16 checks passed**: upload, search, 20 results with images, reasons, filters, sort, product page + similar, catalogue filters + paging, invalid file, admin login (wrong + right password), dashboard, add product (indexed), delete, no console errors |

Retrieval evaluation (`docs/EVALUATION.md`, **cv-baseline on the synthetic fixture**, 300 queries, measured):

| ranking | R@1 | R@5 | R@10 | mAP | p50 latency |
|---|---|---|---|---|---|
| visual only | 69.3% | 88.0% | 93.0% | 0.776 | - |
| hybrid | 71.3% | 90.3% | 94.0% | 0.795 | 45 ms |

Hardest scenarios for the weak baseline: new background (R@1 45%) and lighting (53%). These numbers only show that the
pipeline and metrics work. Real-world accuracy must be measured with FashionCLIP on the Kaggle catalogue.

**Bugs fixed during testing:** pgvector `Vector` type crashed `/similar`; static-file 403 became a 500; React Router
transitions in the UI test; PowerShell `$args` reserved name; 60×80 Kaggle thumbnails rejected by a 64-px minimum; Qwen float32 too large for 16 GB.

## Phase 13: documentation and deployment preparation
* `README.md` (Windows PowerShell, step by step), `docs/API.md`, `docs/DATASET.md`, `docs/MODELS.md`, `docs/EVALUATION.md`, this report, screenshots in `docs/screenshots/`.
* Production: `npm run build` in `frontend`, then `npm run build && npm start` in `backend` serves the built site and API from port 4000.

## Not done / needs your laptop
1. Download the Kaggle dataset and build the real ≥ 1,000-product catalogue (README 1.4).
2. Run FashionCLIP, OpenCLIP and DINOv2 embeddings and `evaluate.py` for real model comparisons (README 3).
3. Optional stages (YOLO, SAM 2, Qwen2.5-VL) need their extra installs and were not executed.
4. DeepFashion2 consumer-to-shop evaluation requires the authors' agreement form; the evaluator accepts any query CSV.


---

# v2 improvement round

See `CHANGELOG.md` for the full list (11 confirmed bugs fixed, features, migrations, files) and
`docs/UPGRADE_V2.md` for applying it to your running installation. Measured in the sandbox
(synthetic fixture, cv-baseline): hybrid R@1 71.3 % -> 75.7 %, mAP 0.795 -> 0.828, exact-match precision kept at 100 %
with twice the recall; tests 41 -> 87 (AI), 13 -> 16 (backend), 16 -> 20 browser checks. FashionCLIP numbers on your
catalogue: run Step 8 of the upgrade guide.
