# Upgrading your existing v2 folder to v3 (garment detection & segmentation)

This copies only **code and docs** into your existing project. Your `.env`, PostgreSQL data, product images,
embeddings, `dataset\` folder, `.venv`, `node_modules`, downloaded models and backups are not touched.

**Nothing has to be recomputed.** Catalogue colour/pattern features and `cv-baseline` vectors still use the
exact v2 foreground method (checked on 252 images: identical masks), and FashionCLIP / OpenCLIP / DINOv2
vectors do not depend on segmentation at all. No database migration is needed.

Commands are for **Windows PowerShell**. Replace the paths with yours.

## 1. Stop the running services

Press `Ctrl + C` in the windows running the AI service, the backend and the frontend.

## 2. Back up the code (30 seconds, lets you undo everything)

```powershell
# a dated copy of your code folders only (not the big dataset / models / node_modules)
$stamp = Get-Date -Format "yyyyMMdd-HHmm"
robocopy C:\Projects\StyleFindAI C:\Projects\StyleFindAI_backup_$stamp /E /XD dataset .venv node_modules .model-cache backups dist __pycache__
```

`robocopy` copies folders; `/E` = include sub-folders, `/XD` = skip these folders.

## 3. Unzip v3 somewhere else and copy it over your project

```powershell
# unzip next to your project (NOT into it)
Expand-Archive C:\Users\you\Downloads\StyleFindAI-v3.zip -DestinationPath C:\Temp\sf3
# copy code + docs over your project; /XF .env makes sure your private settings are never overwritten
robocopy C:\Temp\sf3\StyleFindAI C:\Projects\StyleFindAI /E /XF .env /XD dataset .venv node_modules .model-cache backups
```

robocopy exit codes 0-7 mean success (it reports "copied" / "extra" counts, not errors).

## 4. Settings

Your `.env` keeps working as it is. If it says `GARMENT_ISOLATION=foreground`, the crop now runs the new
pipeline with weight-free methods only. To use the full pipeline, change it to:

```
GARMENT_ISOLATION=auto
```

All new settings have safe defaults (every optional model off). Copy the block starting at
`# Garment isolation used when a search asks for crop=true` from `.env.example` into your `.env` if you want
to see and change them. **Do not copy the whole `.env.example` over your `.env`** (it would replace your
passwords with placeholders).

## 5. Python packages

No new required package. Re-running the install is harmless:

```powershell
cd C:\Projects\StyleFindAI\ai-service
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional models (person masks, parser, SAM 2, pose): `docs/SEGMENTATION.md`, section 3. Recommended first
optional step for photos of people: the 6 MB YOLO person model (section 3.1).

## 6. Check that everything still works

```powershell
cd C:\Projects\StyleFindAI\ai-service
# 1) all Python tests (expect: all passed, 1-2 skipped)
python -m pytest -q
# 2) start the AI service (leave this window open)
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

In a second PowerShell window:

```powershell
cd C:\Projects\StyleFindAI\ai-service
.\.venv\Scripts\Activate.ps1
# 3) health: database ok, same embedding counts as before, outdated_features not higher than before
Invoke-RestMethod http://127.0.0.1:8001/health
# 4) which segmentation models are configured
Invoke-RestMethod http://127.0.0.1:8001/ai/segmentation/status
# 5) segment one of your catalogue photos (replace the file name)
curl.exe -s -F "image=@..\dataset\products\sarees\12345.jpg" http://127.0.0.1:8001/ai/segment
# 6) your FashionCLIP baseline WITHOUT crop: must equal your v2 numbers (hybrid R@1 81.7 %, R@10 95.7 %, mAP 0.870)
python scripts\evaluate.py --models fashion-clip --tag v3_nocrop
# 7) WITH crop, to decide whether "Focus on the garment" helps on your data
python scripts\evaluate.py --models fashion-clip --crop --tag v3_crop
python scripts\compare_experiments.py
```

Then start the backend and frontend as usual (`npm run dev` in `backend` and `frontend`) and run the Node
tests with the AI service running: `cd backend; npx vitest run`.

## 7. Undo

```powershell
robocopy C:\Projects\StyleFindAI_backup_<stamp> C:\Projects\StyleFindAI /E /XF .env
```

Because no data was changed, restoring the code is enough.
