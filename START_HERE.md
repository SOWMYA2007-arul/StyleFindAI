# Start here (Windows PowerShell)

This folder is the complete StyleFind AI project. It already contains your **215 AI-generated dresses** in
`dataset\raw\generated-dresses\`. It does **not** contain your `.env` (passwords) or your old 1,200-product images:
those stay on your laptop and are never put in a zip.

## A. You already have StyleFind running (database, .venv, node_modules, .env)

Copy this folder's contents over your existing project, without touching `.env`:

```powershell
# change both paths; /XF .env = never overwrite your passwords
robocopy "C:\Users\you\Downloads\StyleFindAI" "C:\Projects\StyleFindAI" /E /XF .env
```

Then continue with **C**.

## B. Fresh machine / new folder

```powershell
cd C:\Projects\StyleFindAI
Copy-Item .env.example .env        # then open .env and set the database password
powershell -ExecutionPolicy Bypass -File scripts\check-environment.ps1
```

Then follow README.md section **"1. First-time setup (do this once)"**: database, Python .venv and packages,
`npm install`, migrations.

## C. Import the dresses and run the site

Close the AI service window if it is open, then:

```powershell
cd C:\Projects\StyleFindAI
powershell -ExecutionPolicy Bypass -File scripts\backup-db.ps1
powershell -ExecutionPolicy Bypass -File scripts\import-generated-dresses.ps1 -PreviewOnly
powershell -ExecutionPolicy Bypass -File scripts\import-generated-dresses.ps1
powershell -ExecutionPolicy Bypass -File scripts\start-dev.ps1
```

* `-PreviewOnly` copies nothing; look at `docs\IMPORT_REPORT_AI.md` and `dataset\metadata\ai_preview.jpg`.
* The import takes about 5-10 minutes (FashionCLIP on CPU) and ends with a search-quality report.
* If this folder has no images for your old 1,200 products (case B), show only the dresses:

```powershell
cd ai-service
.\.venv\Scripts\Activate.ps1
python scripts\catalogue_visibility.py --only-source "AI-generated dresses (StyleFind demo)"
```

## D. Optional: more images

* **Up to 540 dresses:** re-run the Colab notebook with the same output folder (docs\GENERATED_DRESSES.md, "Why 215").
* **500 real photos from the CC0 Clothing Dataset:** docs\CLOTHING_DATASET.md.
