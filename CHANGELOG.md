# Changelog

## v3.4 - a colour you type is a requirement (2026-11-26)

Problem: "this dress but in green" also returned a mustard and a beige dress. The typed colour was only 15 % of
the score (visual similarity 65 %), so a dress with the same shape in the wrong colour could outrank green ones; and
for photo-style catalogue images the colour was read from the whole photo, where a beige wall could dominate.

* A colour in the text is now a **requirement**: only products whose colour is in that colour family are shown
  (`green` = green, bottle green, mint, olive, sea green, lime green, teal; shades like "dark green" accept the
  family and rank the closest shade first). Products whose colour is unknown are left out rather than guessed.
* Colour evidence order: verified catalogue colour > measured garment colour (importers) > whole-photo pixel colour.
* A garment type in the text ("dress", "kurta") is also required; with a colour-only request ("this in green") the
  photo's garment type is kept when the AI is at least 55 % sure of it.
* Retrieval: with a typed colour, up to 300 visual candidates plus the 150 visually closest products that have that
  colour, so there are enough results even when the nearest neighbours are all other colours.
* API: new optional `colour_requirement` in `/ai/search` and `/api/search/image` responses
  (`{colour, category, checked, kept, removed}`); the results page says "Only green dresses are shown, as you
  asked" or that none were found. Nothing else in the response changed.
* Tests: 7 new (`tests/test_colour_requirement.py`); checked in a real browser with your request.

## v3.3 - your 215 generated dresses included (2026-11-26)

* `dataset/raw/generated-dresses/`: the 215 SDXL-Turbo dresses you generated, ready to import.
* Generator quality check fixed: v1 required pure-white backgrounds and discarded 323 of 540 good images
  (SDXL-Turbo draws off-white / beige studio walls). Now only dark or cluttered backgrounds are rejected
  (`--strict-white` keeps the old rule; `--no-people` optionally rejects images with a generated model). Re-running
  the notebook with the same folder fills the missing images.
* Importer, generated images: colour = requested colour when it covers >= 8 % of the image centre (92 % of your
  images), else the colour measured on the image centre (the weight-free cut-out often grabbed the beige wall:
  colour agreement went from 56 % to 92 %); requested pattern fills in where the texture classifier cannot tell.
* Tested on your images: 215/215 imported, embedded, evaluated (cv-baseline R@1 0.67 / R@10 0.84) and searched
  through the backend (an edited copy finds the exact dress first).

## v3.2 - AI-generated dress catalogue (2026-11-26)

* `notebooks/generate_dresses_colab.ipynb` + `ai-service/scripts/generate_dress_images.py`: generates e-commerce
  style dress photos (ghost mannequin / flat lay / hanger / dress form, no people) with SDXL-Turbo (fast) or SDXL 1.0
  (quality) from a balanced plan of 540 unique designs; prompts kept within SDXL's 77-token text limit; CLIP quality
  check with regeneration; resumable; output in the importer's layout.
* `scripts/import-generated-dresses.ps1`: preview / import / embed / queries / evaluate for the generated set.
* Importer: `--source`, `--licence`, `--code-prefix`, `--catalogue-csv`; reads generation metadata; names use the
  prompt but the measured colour when the image disagrees; products are labelled AI-generated; separate report files
  per import; fixed sleeve values to the project vocabulary ("short sleeves", "full sleeves").
* Tests: 7 new (`test_generate_dresses.py`). Image realism itself was not testable in the build environment (no GPU).

## v3.1 - Clothing Dataset import (2026-11-25)

* `ai-service/scripts/import_clothing_dataset.py`: imports a diverse subset (default 500) of the CC0
  "Clothing dataset (full, high quality)" (images.csv + images_original + images_compressed). Excludes unclear /
  underwear labels, validates and de-duplicates images, isolates each garment, measures colour and pattern, picks a
  sqrt-proportional quota per garment type with round-robin over pattern groups and farthest-point colour/texture
  sampling. Writes products, report and preview; `--load` adds them to the database (additive, idempotent).
  Measured attributes are stored as unverified `model_prediction` (opt-in `--trust-measured`).
* `scripts/import-clothing-dataset.ps1`: preview / import / embed / queries / evaluate in one command.
* `catalogue_visibility.py`: show only one source or all products, with automatic backup and `--restore`
  (flips `is_active` only; nothing deleted).
* Migration `004_clothing_dataset_categories.sql`: outerwear, pants, shorts, skirts, footwear, hats (insert only).
* `make_eval_queries.py`: `--catalogue` and `--name` so a new catalogue gets its own query set; the baseline
  `queries.csv` is no longer overwritten when testing another catalogue.
* Tests: 6 new (`test_clothing_import.py`); two existing tests made independent of the number of categories /
  products (they assumed exactly 6 categories and only the old catalogue.csv).
* Folder skeleton `dataset/raw/clothing-dataset/` and guide `docs/CLOTHING_DATASET.md`.

## v3.0 - garment detection, segmentation and computer-vision optimisation (2026-11-22)

Scope: AI / computer-vision code only. No login, cart, checkout, order, wishlist or payment code was touched.
Upgrade guide: **docs/UPGRADE_V3.md** (no database migration, nothing to recompute). Full description,
model licences and install commands: **docs/SEGMENTATION.md**.

### What is new
* **Segmentation pipeline** (`ai-service/app/services/segmentation/`) that routes each photo to the cheapest
  adequate method: product photo / person wearing it / several garments / difficult scene. Product photos
  never load a model.
* **Optional models behind one interface**, all off by default, all with fallbacks and timeouts: Ultralytics
  YOLO garment detector (clothing weights) and YOLO-seg person masks, SegFormer human parser, SAM 2 (box +
  positive/negative point prompts, candidate scoring), MediaPipe pose (person photos only).
* **Weight-free improvements**: face/hair/skin removal with a skin model adapted to the wearer's own skin
  tone, multi-garment instance split, hand/occluder removal (pixels removed, never invented; confidence drops),
  guided-filter boundary refinement (thin straps and tassels), black-on-black retry, GrabCut no longer loses
  rows at the image border, outline masks are filled.
* **Structured result** for every photo: path, method, confidence, instances (label, group, bbox, mask,
  occlusion), target, attempts with reasons, fallback used/reason, quality checks, per-stage timings.
* **API**: new `POST /ai/segment`, `GET /ai/segmentation/status`; optional `target` field on `/ai/search`
  (passed through by the Node backend). Existing fields and responses unchanged.
* **Caching**: masks by (image hash, settings, target); query embeddings by (image hash, model, version, crop,
  multi-crop mode).
* **Evaluation**: `scripts/evaluate_segmentation.py` (COCO/Fashionpedia or mask folders; IoU, Dice, precision,
  recall, boundary F1, instance IoU, AP50, mAP, fallback rate, latency, CPU/GPU memory; per category; v1 vs v2
  vs v3 methods) and `scripts/make_segmentation_fixture.py` (260-image synthetic regression set, 13 scene types).
* **Tests**: `tests/test_segmentation.py` (30 tests: routing, every model unavailable/timeout case, fake
  detector/parser/SAM 2, real SegFormer code path with a tiny random model, masks/boxes validity, thin details,
  occlusion, cache, API, search integration, real Ultralytics smoke test when weights are present).

### Bugs found and fixed along the way
| problem | effect | fix |
|---|---|---|
| Ultralytics patches `PIL.Image.open`; for a non-image upload it tries to import a HEIF plugin | with YOLO enabled, a bad upload returned **500** instead of 422 | `load_and_validate` treats that as "not a valid image" |
| DeepFashion2 class `short_sleeve_top` matched the word "short" | a top would have been labelled a lower garment | keyword order + whole-word matching (`cape` no longer rejected as `cap`) |
| GrabCut cannot keep border pixels | garments touching the frame lost their edge rows in the crop | border rows restored from the colour estimate (v3 mode) |
| low-contrast "edges" masks were outlines | garment body whitened in the crop | enclosed outlines are filled (v3 mode) |
| crop used only one colour region of a multi-colour garment | search for close-ups got worse | default crop keeps all garment regions (`ISOLATION_DEFAULT`) |

### Measured in the build sandbox (honest scope)
No licensed real-world annotated dataset was available, so **no real-world segmentation accuracy is claimed**.
Synthetic regression set (260 drawings): IoU 0.879 (v2) → **0.919**, boundary F1 0.833 → **0.881**, strict
2-px boundary F1 0.734 → **0.862**; p50 18 ms weight-free. Retrieval on the sandbox catalogue (cv-baseline,
300 queries): without crop **identical** to v2; with crop hybrid R@1 0.740 → **0.783**, mAP 0.820 → **0.845**,
R@10 0.950 → 0.947, p50 120 → 90 ms; close-up photos got worse (R@1 0.70 → 0.62). One real photo was segmented
with the real YOLO person model. SegFormer, SAM 2, pose and clothing-YOLO weights were **not** run with real
weights (only their code paths, with fakes). Details and limitations: docs/SEGMENTATION.md sections 7-8.

## v2.0 - retrieval correctness, garment understanding, security (2026-11-22)

Upgrade guide for an existing installation: **docs/UPGRADE_V2.md**. All measurements below are from the build
sandbox (360-item synthetic fixture, `cv-baseline` encoder, 300 augmented queries) because FashionCLIP and the
Kaggle data could not be downloaded there. They show the effect of the code changes on one fixed setup; your
FashionCLIP numbers must be measured with the commands in the upgrade guide.

### Bugs found and fixed (confirmed by tests)
| # | Problem | Effect before | Fix |
|---|---|---|---|
| 1 | `SET LOCAL` inside `conn.transaction()` becomes a savepoint when a transaction is open, so planner settings **leaked** into later queries | an exact-scan fallback could silently disable the HNSW index for the rest of a request / an evaluation run; benchmark timings were invalid (150 ms instead of 1 ms) | `scoped_settings()` sets and restores every setting per query |
| 2 | psycopg auto-prepares a statement after 5 runs and PostgreSQL **reuses the cached plan** even when settings or filter values change | fallback queries not actually exact; filter-specific plans ignored | vector queries run with `prepare=False` |
| 3 | `evaluate.py` used k=50 for visual-only but `max(50, SEARCH_CANDIDATES)` for hybrid | candidate-count comparisons inconsistent | `--candidates`, identical N for both |
| 4 | each evaluation **overwrote** `results.json` / `EVALUATION.md` | baseline lost on the next run | one folder per experiment + `compare_experiments.py` (legacy import) |
| 5 | vector results tie-broken by database row order; OpenCV k-means/GrabCut used a global RNG | same photo could give a different order (measured `stable=False`) | deterministic sort keys + seeded RNG; stability check in every experiment |
| 6 | exact-match: pHash ignores colour and fine detail | the same garment with stripes drawn on it was labelled "exact" | colour check + SSIM check + uniqueness rule (thresholds measured) |
| 7 | image replacement saved the new file before the DB transaction | orphan files when the update failed | clean-up on rollback; DB trigger invalidates embeddings when an image changes |
| 8 | failed re-indexing after an image change was invisible | product visible but not searchable | `index_status` / `index_error`, admin failures panel with Retry |
| 9 | text parser: dictionary merge dropped `maroon`; typo fixer turned "sleeves" into "sleeveless" | wrong colour / sleeve filters from ordinary text | vocabulary rebuilt, known words protected, tests for every example in the brief |
| 10 | every page load called `/api/auth/me` | a 401 console error for every shopper | public `/api/auth/session`, checked only on admin pages |
| 11 | partial (close-up) photos: the garment touching the borders was treated as background | wrong colour/pattern features for close-ups | "fills frame" detection |

### Before / after (sandbox, cv-baseline, same 300 queries, default settings)
| metric | v1 code | v2 code |
|---|---|---|
| hybrid Recall@1 | 71.3 % | **75.7 %** |
| hybrid Recall@10 | 94.0 % | **95.7 %** |
| hybrid mAP | 0.795 | **0.828** |
| visual-only Recall@1 | 69.3 % | 72.3 % |
| exact-match precision / recall | 100 % / 6.3 % | **100 % / 13.7 %** |
| pattern consistency@10 | 30.0 % | 36.6 % |
| colour consistency@10 | 72.9 % | 74.2 % |
| latency p50 / p95 | 45 / 59 ms | 59-63 / 77-81 ms |

By scenario (hybrid R@1): background 45 -> 60 %, clutter 65 -> 83 %, angle 97 -> 98 %, lighting 53 -> 57 %,
**partial 97 -> 80 % (regression)**. The partial regression is specific to `cv-baseline`, whose features use the garment
mask; FashionCLIP embeds the full image, so for it only the colour/pattern re-ranking components change.

Experiments (all stored, `docs/sandbox_results_v2/EXPERIMENTS.md`):
* candidates 50 / 100 / 200: identical accuracy, p50 59 / 68 / 84 ms -> keep **50** on this data
* ef_search 40 / 100 / 400: identical (PostgreSQL uses exact scans at this size; see index benchmark)
* garment crop: R@1 75.7 -> 74.0 %, latency x2 -> stays opt-in
* multi-crop max-score fusion: R@1 **82.7 %**, mAP 0.883, +40 ms; mean-embedding: 78.0 % -> worth testing with FashionCLIP
* weight tuning on val, checked on test: best-on-val weights did **not** improve test -> keep current weights
* foreground masks vs ground truth (IoU): white-on-white 0.61 -> 0.88 (0.94 with GrabCut), shadow 0.80 -> 0.97,
  gradient backdrop 0.61 -> 0.99, clutter 0.88 -> 0.97

### Features
* Retrieval: configurable `HNSW_EF_SEARCH`, pgvector >= 0.8 iterative scans, exact fallback for selective filters,
  price filters, only verified attributes can filter, inactive/non-primary images never returned, `retrieval` info and
  per-stage timings in every response (`preprocess`, `embed`, `attributes`, `vector_search`, `details`, `rerank`,
  `backend_to_ai_roundtrip`, `backend_total`), `Server-Timing` header, request ids across services.
* Garment isolation: multi-cue foreground (two-cluster backdrop, shadows, edges for white-on-white, main region, nested
  photo, close-ups, GrabCut), `GARMENT_ISOLATION=foreground|yolo|off`, padding so sleeves/hems/borders survive, fallback chain.
* Colour: CIEDE2000, 41 reference colours incl. maroon/wine/burgundy/royal blue/cream/mustard/olive/bottle green/mint,
  shade merging, secondary colour only when it is a real share, multicolour flag, "darker/lighter" requests.
* Pattern: texture classifier (solid / striped / checked / polka dot, with confidence and "unknown"), merged with
  zero-shot classes (floral, paisley, geometric, abstract, ethnic motifs, embroidered, textured).
* Attributes (zero-shot, structured value/confidence/source): + length, silhouette, fabric appearance, embroidery, border.
* Text: synonyms and Indian-wear spellings (kurthi, sari, lehnga, ghagra ...), typo correction, negation
  ("without embroidery"), category / length / fabric / shade.
* Model management: `ModelUnavailableError` -> HTTP 503 with a clear message and 60 s cooldown, dimension/version check
  after loading, optional revision pinning (`FASHION_CLIP_REVISION`, new `model_version`, never mixed), `cv-baseline`
  bumped to `cv-v2`.
* Evaluation: `--candidates`, `--ef-search`, `--crop`, `--multicrop`, `--split val|test`, `--weights`, metadata (dataset /
  query-set hashes, hardware, pgvector, RSS memory), stability check, `compare_experiments.py`, `tune_weights.py`,
  `benchmark_vector_index.py`, `benchmark_foreground.py`, `inspect_dataset.py` (read-only), `recompute_features.py`,
  optional `occlusion` queries and a product-level `split` column.
* Security: HttpOnly + SameSite=Strict session cookie, CSRF double-submit token, logout revokes all tokens
  (`token_version`), Bearer mode kept for scripts, upload extension/dimension/pixel checks, production refuses weak secrets.
* Admin: indexing failures panel with Retry, per-product re-index, show/hide toggle, accessible confirmation dialog.
* Shop: results filters (category, colour, pattern, sleeves, neckline, price) re-run the search **on the server**,
  price sorting, product image gallery.
* Ops: `backup-db.ps1`, `restore-db.ps1`, `start-prod.ps1`, `migrate.py --rollback`.

### Database migrations
| file | change | rollback |
|---|---|---|
| `003_integrity_and_index_status.sql` | `product_images.index_status/index_error/indexed_at` (backfilled), trigger deleting embeddings + features when an image file changes, trigger marking images indexed, `admin_users.token_version`, `cv-v2` model row, non-blank product name check (new rows) | `down/003_integrity_and_index_status.down.sql` |
| `down/002_vector_indexes.down.sql` | rollback script for the existing 002 (drops HNSW indexes only) | - |

### Files
**Modified (50):** `.env.example`; ai-service: `app/api/routes.py`, `app/config.py`, `app/main.py`,
`app/models/encoders.py`, `app/schemas/api.py`, `app/services/{attributes,colour,detection,foreground,indexing,pattern,
ranking,search,vector_store}.py`, `app/utils/keywords.py`, `requirements.txt`, `scripts/{evaluate,make_eval_queries,
migrate}.py`; backend: `package.json`, `package-lock.json`, `src/app.ts`, `src/config/env.ts`,
`src/controllers/{adminController,publicController,validators}.ts`, `src/middleware/{auth,upload}.ts`,
`src/routes/index.ts`, `src/services/{adminService,aiClient,productService,searchService}.ts`, `tests/api.test.ts`;
`database/schema.sql`; frontend: `src/components/{ProtectedRoute,ResultCard}.tsx`,
`src/context/{AuthContext,SearchContext}.tsx`, `src/layouts/AdminLayout.tsx`,
`src/pages/{AdminDashboardPage,AdminLoginPage,ProductDetailsPage,ProductManagementPage,SearchResultsPage}.tsx`,
`src/services/api.ts`, `src/types/index.ts`; `tests/ui-smoke.mjs`; docs (README, API, MODELS, DATASET, PHASE_REPORT).

**New:** `ai-service/scripts/{benchmark_foreground,benchmark_vector_index,compare_experiments,inspect_dataset,
recompute_features,tune_weights}.py`, `ai-service/tests/{test_exact_match,test_improvements}.py`,
`database/migrations/003_integrity_and_index_status.sql`, `database/migrations/down/*.down.sql`,
`frontend/src/components/ConfirmDialog.tsx`, `scripts/{backup-db,restore-db,start-prod}.ps1`, `CHANGELOG.md`,
`docs/{UPGRADE_V2,DEPLOYMENT,SHOPPING_EXTENSION}.md`, `docs/sandbox_results_v2/*`.

**Unchanged on purpose:** `001_init.sql`, `002_vector_indexes.sql`, `db.py`, `preprocess.py`, `prepare_dataset.py`,
`validate_images.py`, `remove_duplicates.py`, `load_catalogue.py`, `generate_embeddings.py`, catalogue / product /
category routes, layouts, home / catalogue / about pages.

### Tests
| suite | v1 | v2 |
|---|---|---|
| AI service (pytest) | 41 passed, 1 skipped | **87 passed**, 1 skipped (the >= 1000-products rule, fixture only) |
| Backend (vitest, real DB + AI) | 13 passed | **16 passed** |
| Browser end-to-end (Playwright) | 16 checks | **20 checks**, 0 console errors |
| Type checks / production builds | clean | clean |

### Known limitations / next steps
* FashionCLIP / OpenCLIP / DINOv2 comparisons and all real-catalogue numbers still have to be run on your laptop.
* Queries are augmented catalogue photos; independent customer photos (DeepFashion2 consumer images under its licence,
  or your own photos) are needed for real-world accuracy - `evaluate.py --queries <csv> --real-photos`.
* A person wearing a garment stays in the `foreground` crop; separating clothing from the person needs a clothing
  detector/segmenter (YOLO trained on clothing classes or SAM 2), which remain optional and untested here.
* Pattern classification on 60x80 images is weak by nature (confidence is reduced, mostly "unknown").
* Composed image+text search is still a training-free baseline.
* Customer accounts / cart / checkout are designed (docs/SHOPPING_EXTENSION.md) but not implemented.
