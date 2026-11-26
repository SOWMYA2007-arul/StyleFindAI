# Imports a diverse subset of the Clothing Dataset (CC0) into StyleFind, step by step.
#
#   powershell -ExecutionPolicy Bypass -File scripts\import-clothing-dataset.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\import-clothing-dataset.ps1 -Count 500 -Model fashion-clip
#   powershell -ExecutionPolicy Bypass -File scripts\import-clothing-dataset.ps1 -PreviewOnly
#
# Before running: PostgreSQL is running, and the dataset is in dataset\raw\clothing-dataset\
# (images.csv + images_original\ + images_compressed\). Back up first: scripts\backup-db.ps1
# Nothing is deleted. Stop the AI service first if it is running (it shares the model RAM).
param(
    [int]$Count = 500,
    [string]$Model = "fashion-clip",
    [string]$Raw = "",
    [switch]$PreviewOnly
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if ($Raw -eq "") { $Raw = Join-Path $root "dataset\raw\clothing-dataset" }

if (-not (Test-Path (Join-Path $Raw "images.csv"))) {
    Write-Host "images.csv not found in $Raw" -ForegroundColor Red
    Write-Host "Copy images.csv, images_original and images_compressed into that folder (see docs\CLOTHING_DATASET.md)."
    exit 1
}
Set-Location (Join-Path $root "ai-service")
. .\.venv\Scripts\Activate.ps1

function Step($text, [scriptblock]$cmd) {
    Write-Host "`n=== $text ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "Step failed: $text" -ForegroundColor Red; exit $LASTEXITCODE }
}

Step "1/6 Database migrations (adds the new categories)" { python scripts\migrate.py }
if ($PreviewOnly) {
    Step "Preview: choose $Count images, copy nothing" { python scripts\import_clothing_dataset.py --raw "$Raw" --count $Count --pattern-model $Model --dry-run }
    Write-Host "`nOpen docs\CLOTHING_DATASET_IMPORT.md and dataset\metadata\clothing_dataset_preview.jpg, then run without -PreviewOnly."
    exit 0
}
Step "2/6 Choose $Count diverse images, copy them, add them to the database" { python scripts\import_clothing_dataset.py --raw "$Raw" --count $Count --pattern-model $Model --load }
Step "3/6 Embeddings for the new images only ($Model)" { python scripts\generate_embeddings.py --model $Model }
Step "4/6 Test queries for the new catalogue (your old baseline queries are kept)" { python scripts\make_eval_queries.py --catalogue ..\dataset\metadata\clothing_dataset_catalogue.csv --name queries_clothing --n 100 }
Step "5/6 Evaluate without and with garment crop" {
    python scripts\evaluate.py --models $Model --queries ..\dataset\evaluation\queries_clothing.csv --tag clothing_nocrop
    if ($LASTEXITCODE -eq 0) { python scripts\evaluate.py --models $Model --queries ..\dataset\evaluation\queries_clothing.csv --crop --tag clothing_crop }
    if ($LASTEXITCODE -eq 0) { python scripts\compare_experiments.py }
}
Step "6/6 What is visible on the website now" { python scripts\catalogue_visibility.py --status }
Write-Host "`nDone. Start the site with scripts\start-dev.ps1. Report: docs\CLOTHING_DATASET_IMPORT.md" -ForegroundColor Green
Write-Host "To show ONLY the new products: python scripts\catalogue_visibility.py --only-source `"clothing-dataset-full (Alexey Grigorev)`"  (undo: --restore)"
