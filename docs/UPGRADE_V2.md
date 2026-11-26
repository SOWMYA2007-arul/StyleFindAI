# Upgrading your running StyleFindAI to v2 (Windows PowerShell)

This keeps your `.env`, your 1,200-product catalogue, images, FashionCLIP embeddings and your evaluation
baseline. Do the steps **in order** and check the "you should see" line before moving on. If anything looks
different, stop and send me the output.

What v2 changes in your data:
* adds columns / triggers with migration `003` (no rows deleted, rollback script included)
* recomputes the colour/pattern features table (`image_features`) - derived data, **embeddings are not touched**
* nothing else. No catalogue replacement, no embedding regeneration.

---

## Step 1 - Back up (safe, read-only for your data)

Folder: `D:\Projects\StyleFindAI` (Docker Desktop running)

```powershell
cd D:\Projects\StyleFindAI
docker compose up -d
mkdir backups -Force
Copy-Item dataset\evaluation\results.json backups\results_fashionclip_baseline.json
Copy-Item docs\EVALUATION.md backups\EVALUATION_fashionclip_baseline.md
docker exec stylefind-db pg_dump -U postgres -d stylefind -Fc -f /tmp/before_v2.dump
docker cp stylefind-db:/tmp/before_v2.dump backups\before_v2.dump
```
*Why the two-step dump:* PowerShell's `>` re-encodes binary output and would corrupt the backup.

You should see: `backups\before_v2.dump` with a size of several MB (`Get-ChildItem backups`).

Also copy your whole project folder once (code only is enough), e.g. to `D:\Projects\StyleFindAI_v1_backup`, so you
can roll back the code:
```powershell
robocopy D:\Projects\StyleFindAI D:\Projects\StyleFindAI_v1_backup /E /XD node_modules .venv .model-cache dataset
```

## Step 2 - Copy the v2 code over your project (keeps .env and data)

Unzip `StyleFindAI-v2.zip` somewhere else first, e.g. `D:\Downloads\StyleFindAI-v2\StyleFindAI`. Then:

```powershell
robocopy D:\Downloads\StyleFindAI-v2\StyleFindAI D:\Projects\StyleFindAI /E /XD node_modules .venv .model-cache dataset backups /XF .env
```
* `/E` copies all folders, `/XD` skips your data/dependency folders, `/XF .env` never touches your secrets.
* robocopy does **not** delete files that exist only on your side (no `/MIR`).
* robocopy exit codes 0-7 mean success (it reports "copied"/"extra" files); 8+ is an error.

You should see: a summary table with `FAILED 0`.

## Step 3 - Install the new dependencies

```powershell
cd D:\Projects\StyleFindAI\ai-service
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # adds psutil (memory measurement)
cd ..\backend;  npm install              # adds cookie-parser (secure admin login)
cd ..\frontend; npm install
```
You should see: no red `ERROR` lines.

## Step 4 - Database migration 003

```powershell
cd D:\Projects\StyleFindAI\ai-service
.\.venv\Scripts\Activate.ps1
python scripts/migrate.py --status
python scripts/migrate.py
```
You should see: `applying 003_integrity_and_index_status.sql ... ok` and your pgvector version.

Check that nothing was lost (should print 1200 / 1200 / your embedding count):
```powershell
docker exec stylefind-db psql -U postgres -d stylefind -c "select count(*) from products" -c "select index_status, count(*) from product_images group by 1" -c "select model_key, model_version, count(*) from image_embeddings group by 1,2"
```

## Step 5 - Recompute colour / pattern features (embeddings untouched)

The improved garment mask, CIEDE2000 colours and pattern classifier must describe catalogue images exactly like
query images:
```powershell
python scripts/recompute_features.py
```
You should see: `updated 1200 images ... (embeddings unchanged)` - about 1-2 minutes on a laptop CPU.

## Step 6 - Start and check

```powershell
cd D:\Projects\StyleFindAI
powershell -ExecutionPolicy Bypass -File scripts\start-dev.ps1
```
Open http://127.0.0.1:8001/health - you should see `"outdated_features": 0` and your embedding counts.
Open http://localhost:5173 and run a search. Log in to the admin again (sessions now use a secure cookie, old
tokens from v1 are no longer used).

## Step 7 - Keep your FashionCLIP baseline in the comparisons

```powershell
cd D:\Projects\StyleFindAI\ai-service
.\.venv\Scripts\Activate.ps1
python scripts/compare_experiments.py --import-legacy ..\backups\results_fashionclip_baseline.json --label fashionclip-baseline-v1
```
You should see: `imported legacy_fashion-clip_fashionclip-baseline-v1` (your 81.7 % R@1 run).

## Step 8 - The Priority-1 experiments (same queries, same model, same DB)

Keep your existing `dataset\evaluation\queries.csv` - do **not** regenerate it, so results stay comparable with
the baseline. Each command stores its own experiment folder; nothing is overwritten.

```powershell
# candidate count: 50 vs 100 vs 200 (visual-only and hybrid both use the same N now)
python scripts/evaluate.py --models fashion-clip --candidates 50 100 200 --tag candidates
# HNSW ef_search (no model needed, ~1 min): plans, recall vs exact, filters
python scripts/benchmark_vector_index.py --models fashion-clip --k 50 100 200 --ef 40 64 100 200 400
# optional quality experiments
python scripts/evaluate.py --models fashion-clip --multicrop max_score --tag multicrop
python scripts/evaluate.py --models fashion-clip --crop --tag crop
python scripts/tune_weights.py --model fashion-clip
# side-by-side report -> docs\EXPERIMENTS.md
python scripts/compare_experiments.py
```
Send me `docs\EXPERIMENTS.md`, `docs\INDEX_BENCHMARK.md` and `docs\WEIGHT_TUNING.md` and we choose settings from
YOUR numbers (I expect the candidate count to make little difference at 1,200 products, but we measure it).

## Step 9 - Inspect the new high-resolution dataset (read-only)

```powershell
python scripts/inspect_dataset.py --root "D:\path\to\new-dataset"
```
It writes `docs\NEW_DATASET_REPORT.md` (resolution, categories and their mapping to StyleFind, duplicates,
corrupt files, licence text). Nothing is imported. Send me the report before we plan integration.

---

## Rolling back

* Database to the state before v2: `powershell -ExecutionPolicy Bypass -File scripts\restore-db.ps1 -File backups\before_v2.dump`
* Only migration 003: `python scripts/migrate.py --rollback 003_integrity_and_index_status.sql`
* Code: copy `D:\Projects\StyleFindAI_v1_backup` back the same way (robocopy, excluding .env and dataset).
