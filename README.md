# StyleFind AI

AI-powered visual fashion search. Upload a photo of a kurta, saree, lehenga or dress and StyleFind:

1. reads and validates the photo,
2. turns it into an **embedding** (a list of 512 numbers) with **FashionCLIP**,
3. searches every catalogue embedding stored in **PostgreSQL + pgvector**,
4. re-ranks the 50 closest products by visual look, colour, pattern and attributes,
5. returns real product records and shows their stored images: an exact match if one exists, then similar and related pieces.

No search result is hard-coded or mocked. Every result comes from the database through the model.

> **New here or upgrading? Read [START_HERE.md](START_HERE.md) first.** Your 215 AI-generated dresses are already in
> `dataset/raw/generated-dresses/`.
>
> **Want 500 Amazon-style AI-generated dress photos?** Generate them on free Google Colab with
> `notebooks/generate_dresses_colab.ipynb`, then follow **[docs/GENERATED_DRESSES.md](docs/GENERATED_DRESSES.md)**.
>
> **Adding the Clothing Dataset (500 diverse photos)?** Put it in `dataset/raw/clothing-dataset/` and follow
> **[docs/CLOTHING_DATASET.md](docs/CLOTHING_DATASET.md)** (one PowerShell command).
>
> **Already running v2?** Follow **[docs/UPGRADE_V3.md](docs/UPGRADE_V3.md)** (garment detection & segmentation
> upgrade; nothing in your database has to be recomputed). From v1: **[docs/UPGRADE_V2.md](docs/UPGRADE_V2.md)** first.
> What changed and the measured before/after: **[CHANGELOG.md](CHANGELOG.md)**. Segmentation details:
> **[docs/SEGMENTATION.md](docs/SEGMENTATION.md)**.

```
React (Vite, :5173) ──► Node/Express API (:4000) ──► Python FastAPI AI service (:8001)
                              │                              │  FashionCLIP / OpenCLIP / DINOv2
                              └──────── PostgreSQL 16 + pgvector ◄──┘  (HNSW cosine index)
```

| Folder | What it contains |
|---|---|
| `ai-service/` | Python 3.11 FastAPI service: preprocessing, encoders, pgvector search, hybrid ranking, attributes, dataset + evaluation scripts, pytest tests |
| `backend/` | Node 22 + Express + TypeScript REST API: uploads, products, admin auth, talks to the AI service |
| `frontend/` | React + TypeScript + Vite + Tailwind website |
| `database/` | `schema.sql` and ordered `migrations/` |
| `dataset/` | `raw/` (downloads), `products/` (clean images by category), `metadata/catalogue.csv`, `evaluation/` |
| `docs/` | API reference, dataset notes, model notes, evaluation report, phase report, screenshots |
| `scripts/` | Windows PowerShell helpers (`check-environment.ps1`, `start-dev.ps1`) |
| `tests/` | Browser end-to-end test (`ui-smoke.mjs`, Playwright) |

---

## 0. What you need

| Tool | Version | Check in PowerShell |
|---|---|---|
| Python | 3.11.x | `py -3.11 --version` |
| Node.js | 22 LTS (20+ works) | `node --version` |
| Docker Desktop (recommended) | any recent | `docker --version` |
| Disk / RAM | ~6 GB free, 16 GB RAM | |

Run the automatic check (it only reads your system and changes nothing):

```powershell
cd D:\Projects\StyleFindAI
powershell -ExecutionPolicy Bypass -File scripts\check-environment.ps1
```

> **Why Docker for the database?** pgvector (the vector-search extension) has no simple Windows
> installer; building it needs Visual Studio C++ tools. The `pgvector/pgvector:pg16` Docker image
> already contains it. If you prefer native PostgreSQL, see "Native PostgreSQL" at the end.

---

## 1. First-time setup (do this once)

All commands are typed in **PowerShell**. Lines starting with `#` are explanations, not commands.

### 1.1 Configuration file

```powershell
cd D:\Projects\StyleFindAI
# make your private copy of the settings (never commit .env)
Copy-Item .env.example .env
notepad .env
```

In Notepad change at least `POSTGRES_PASSWORD`, `JWT_SECRET` (any long random text) and `AI_INTERNAL_TOKEN`, then save.

### 1.2 Start the database

```powershell
# downloads PostgreSQL+pgvector the first time, then runs it in the background
docker compose up -d
# should say "healthy" after ~10 seconds
docker compose ps
```

### 1.3 Python AI service

```powershell
cd ai-service
# create an isolated Python environment in the folder .venv
py -3.11 -m venv .venv
# activate it (your prompt now starts with (.venv))
.\.venv\Scripts\Activate.ps1
# if PowerShell blocks the script, run once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
# PyTorch CPU build first: much smaller than the default CUDA build, you have no NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# create the tables, indexes and categories in PostgreSQL
python scripts/migrate.py
```

### 1.4 Get the dataset (real product images)

1. Sign in to Kaggle and download **Fashion Product Images (Small)**:
   https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small (about 590 MB zip).
2. Unzip it to `D:\Projects\StyleFindAI\dataset\raw\fashion-product-images-small\`
   so that `styles.csv` and the `images\` folder are inside (a nested `myntradataset\` folder is fine too).

Then, still inside `ai-service` with `.venv` active:

```powershell
# filter women's clothing, remove footwear/bags/jewellery/innerwear/etc., check images,
# remove duplicates, pick per-category quotas, copy into dataset\products\<category>\<id>.jpg
python scripts/prepare_dataset.py --raw dataset/raw/fashion-product-images-small
# verify every image and metadata row (must print PASSED)
python scripts/validate_images.py --min-count 1000
# counts per category / colour / pattern -> docs\DATASET_STATISTICS.md
python scripts/dataset_statistics.py
# put products into PostgreSQL
python scripts/load_catalogue.py
```

`prepare_dataset.py` prints a report. The Kaggle set has far fewer than 150 lehengas, so that
quota is filled from other categories automatically; see `docs/DATASET.md` for the reasons and for legitimate ways to add more.

### 1.5 Generate embeddings (the AI "index")

```powershell
# downloads FashionCLIP (~600 MB, once) and embeds every catalogue image on the CPU.
# Only images without an embedding are processed, so it is safe to re-run.
python scripts/generate_embeddings.py --model fashion-clip
```

On a 16 GB laptop CPU this typically takes a few minutes for ~1,200 images (progress and img/s are printed).
For heavy experiments use Google Colab: set `AI_DEVICE=cuda` in `.env` there and run the same script.

### 1.6 PROOF: the AI service searches the real dataset (before any website)

```powershell
# start the AI service (leave this window open)
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Open a **second** PowerShell window:

```powershell
cd D:\Projects\StyleFindAI\ai-service
.\.venv\Scripts\Activate.ps1
# make test photos with known answers (new background, angle, crop, clutter, lighting)
python scripts/make_eval_queries.py --n 60
# health: database ok, pgvector version, number of embeddings per model
curl.exe http://127.0.0.1:8001/health
# upload one of those photos to the real search endpoint
$q = (Get-ChildItem ..\dataset\evaluation\queries\*__background.jpg | Select-Object -First 1).FullName
curl.exe -F "image=@$q" -F "top_k=5" http://127.0.0.1:8001/ai/search
```

The file name starts with the correct product id (e.g. `fpi-15970__background.jpg`); it should
appear at or near the top of `results`. Interactive API docs: http://127.0.0.1:8001/docs

### 1.7 Backend and frontend

```powershell
cd D:\Projects\StyleFindAI\backend
npm install
# create your admin login (choose your own password, 10+ characters)
npm run create-admin -- --email admin@stylefind.local --password "choose-a-long-password"

cd ..\frontend
npm install
```

---

## 2. Run the application (every time)

```powershell
cd D:\Projects\StyleFindAI
docker compose up -d
powershell -ExecutionPolicy Bypass -File scripts\start-dev.ps1
```

This opens three windows (AI service, backend, frontend). Then open **http://localhost:5173**.

- Admin: http://localhost:5173/admin/login
- Backend health: http://localhost:4000/api/health
- AI docs: http://127.0.0.1:8001/docs

To stop: close the three windows, then `docker compose stop`.

---

## 3. Tests

```powershell
# Python: unit tests + dataset tests + API/integration tests incl. the 8 AI scenarios
cd ai-service; .\.venv\Scripts\Activate.ps1; python -m pytest tests -v

# Retrieval evaluation: every run is stored separately in dataset\evaluation\experiments\
python scripts/evaluate.py --models fashion-clip --tag baseline
python scripts/evaluate.py --models fashion-clip --candidates 50 100 200 --tag candidates
python scripts/generate_embeddings.py --model openclip-b32 --no-features
python scripts/generate_embeddings.py --model dinov2-b14 --no-features
python scripts/evaluate.py --models fashion-clip openclip-b32 dinov2-b14 --tag models
python scripts/compare_experiments.py              # -> docs\EXPERIMENTS.md
python scripts/benchmark_vector_index.py           # HNSW plans / recall / ef_search -> docs\INDEX_BENCHMARK.md
python scripts/tune_weights.py --model fashion-clip # val/test weight tuning -> docs\WEIGHT_TUNING.md
python scripts/inspect_dataset.py --root <folder>  # READ-ONLY check of a new dataset

# Node API integration tests (AI service must be running)
cd ..\backend; npm test

# Browser end-to-end test (all three services running)
cd ..\tests; npm install; npx playwright install chromium
node ui-smoke.mjs --query ..\dataset\evaluation\queries\<a query file>.jpg --email admin@stylefind.local --password "<your password>"
```

---

## 4. Configuration highlights (`.env`)

| Variable | Meaning |
|---|---|
| `AI_PRIMARY_MODEL` | encoder used for search: `fashion-clip` (default), `openclip-b32`, `dinov2-b14`, `cv-baseline` |
| `RANK_W_VISUAL / COLOUR / PATTERN / ATTRIBUTE` | hybrid ranking weights (0.65 / 0.15 / 0.10 / 0.10), normalised automatically |
| `SEARCH_CANDIDATES` | how many pgvector neighbours are re-ranked (50) |
| `ENABLE_GARMENT_DETECTION`, `ENABLE_SAM2`, `ENABLE_VLM_ATTRIBUTES` | optional heavy stages, off by default (see `docs/MODELS.md`) |
| `STORE_SEARCH_HISTORY` | anonymous logging: only an image hash and predicted attributes, never the photo |
| `HNSW_EF_SEARCH`, `HNSW_ITERATIVE_SCAN`, `FILTERED_EXACT_FALLBACK` | vector-index behaviour (defaults are safe; see docs/INDEX_BENCHMARK.md after running the benchmark) |
| `SEARCH_MULTICROP` | `off` (default) / `max_score` / `mean_embedding` - enable only after `evaluate.py --multicrop` shows a gain |
| `GARMENT_ISOLATION`, `ISOLATION_PADDING`, `ISOLATION_DEFAULT` | what "Focus on the garment" does: `auto` (segmentation pipeline, default), `foreground` (weight-free only), `yolo`, `off` |
| `PERSON_SEGMENTER`, `HUMAN_PARSER`, `GARMENT_DETECTOR`, `ENABLE_SAM2`, `POSE_ESTIMATOR` | optional segmentation models, all off by default (see `docs/SEGMENTATION.md`) |
| `FASHION_CLIP_REVISION` | optional model pinning (empty keeps existing embeddings valid) |
| `AI_MAX_LOADED_MODELS` | how many encoders may stay in RAM at once (1 on a 16 GB laptop) |

---

## 5. Integrating with ThreadKart later

The Python service is self-contained and talks plain HTTP/JSON (`docs/API.md`). ThreadKart can either
call the Node endpoint `POST /api/search/image` or call `POST /ai/search` directly, and load its own
catalogue with the same scripts (`load_catalogue.py`, `generate_embeddings.py`). StyleFind does not
read or modify any ThreadKart files.

---

## Backups, production, security

* Back up: `powershell -ExecutionPolicy Bypass -File scripts\backup-db.ps1` (restore: `scripts\restore-db.ps1 -File ...`).
* Production-style start (built site + API on one port): `scripts\start-prod.ps1`; checklist and HTTPS in
  [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
* Admin sessions use an HttpOnly cookie + CSRF token; logout revokes every token. Scripts can still log in with
  `POST /api/auth/login?mode=bearer`.
* Undo a migration: `python scripts/migrate.py --rollback <file>` (asks first; data-preserving).

## Native PostgreSQL (without Docker)

1. Install PostgreSQL 16 from postgresql.org.
2. Build pgvector following https://github.com/pgvector/pgvector#windows (needs "x64 Native Tools Command Prompt for VS").
3. Create the database: `psql -U postgres -c "CREATE DATABASE stylefind;"` and set the same credentials in `.env`.
4. Continue with `python scripts/migrate.py`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Website says "AI search service is not running" | start `uvicorn` in `ai-service` (step 1.6) |
| Search returns HTTP 409 "No catalogue embeddings" | run `generate_embeddings.py` for that model |
| `connection refused` on port 5432 | `docker compose up -d`, wait for "healthy" |
| First search is slow | the model loads on first use (~5-15 s), later searches are fast |
