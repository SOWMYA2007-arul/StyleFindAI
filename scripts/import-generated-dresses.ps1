# Imports the AI-generated dress photos (made with notebooks\generate_dresses_colab.ipynb) into StyleFind.
#
#   powershell -ExecutionPolicy Bypass -File scripts\import-generated-dresses.ps1 -PreviewOnly
#   powershell -ExecutionPolicy Bypass -File scripts\import-generated-dresses.ps1
#
# Before running: PostgreSQL is running, the AI service window is closed, and the unzipped images are in
# dataset\raw\generated-dresses\ (images.csv + images_original\). Back up first: scripts\backup-db.ps1
# Nothing is deleted; the products are marked as AI-generated.
param(
    [int]$Count = 500,
    [string]$Model = "fashion-clip",
    [string]$Raw = "",
    [switch]$PreviewOnly
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if ($Raw -eq "") { $Raw = Join-Path $root "dataset\raw\generated-dresses" }
$Source = "AI-generated dresses (StyleFind demo)"
$Licence = "AI-generated with Stable Diffusion (SDXL-Turbo: Stability AI Community License / SDXL 1.0: CreativeML Open RAIL++-M)"

if (-not (Test-Path (Join-Path $Raw "images.csv"))) {
    Write-Host "images.csv not found in $Raw" -ForegroundColor Red
    Write-Host "Unzip generated-dresses.zip from the Colab notebook into that folder (see docs\GENERATED_DRESSES.md)."
    exit 1
}
Set-Location (Join-Path $root "ai-service")
. .\.venv\Scripts\Activate.ps1

function Step($text, [scriptblock]$cmd) {
    Write-Host "`n=== $text ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "Step failed: $text" -ForegroundColor Red; exit $LASTEXITCODE }
}
$common = @("--raw", $Raw, "--count", $Count, "--min-per-type", "0", "--pattern-model", $Model, "--code-prefix", "ai",
            "--source", $Source, "--licence", $Licence,
            "--catalogue-csv", (Join-Path $root "dataset\metadata\generated_dresses_catalogue.csv"))

Step "1/6 Database migrations" { python scripts\migrate.py }
if ($PreviewOnly) {
    Step "Preview: choose $Count images, copy nothing" { python scripts\import_clothing_dataset.py @common --dry-run }
    Write-Host "`nOpen docs\IMPORT_REPORT_AI.md and dataset\metadata\ai_preview.jpg, then run without -PreviewOnly."
    exit 0
}
Step "2/6 Check, choose and add $Count dresses to the database" { python scripts\import_clothing_dataset.py @common --load }
Step "3/6 Embeddings for the new images only ($Model)" { python scripts\generate_embeddings.py --model $Model }
Step "4/6 Test queries for the new dresses (your baseline queries are kept)" { python scripts\make_eval_queries.py --catalogue ..\dataset\metadata\generated_dresses_catalogue.csv --name queries_generated --n 100 }
Step "5/6 Evaluate without and with garment crop" {
    python scripts\evaluate.py --models $Model --queries ..\dataset\evaluation\queries_generated.csv --tag generated_nocrop
    if ($LASTEXITCODE -eq 0) { python scripts\evaluate.py --models $Model --queries ..\dataset\evaluation\queries_generated.csv --crop --tag generated_crop }
    if ($LASTEXITCODE -eq 0) { python scripts\compare_experiments.py }
}
Step "6/6 What is visible on the website now" { python scripts\catalogue_visibility.py --status }
Write-Host "`nDone. Start the site with scripts\start-dev.ps1. Report: docs\IMPORT_REPORT_AI.md" -ForegroundColor Green
Write-Host "To show ONLY the generated dresses: python scripts\catalogue_visibility.py --only-source `"$Source`"  (undo: --restore)"
