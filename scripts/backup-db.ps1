# Back up the StyleFind database (products, images metadata, embeddings, admin users) to backups\.
#   powershell -ExecutionPolicy Bypass -File scripts\backup-db.ps1
# Image FILES are not inside the database: back up dataset\products separately (copy the folder).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$dir = Join-Path $root "backups"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$file = "stylefind_$stamp.dump"
# Write the dump INSIDE the container, then copy it out (PowerShell '>' would corrupt binary output).
docker exec stylefind-db pg_dump -U postgres -d stylefind -Fc -f "/tmp/$file"
if ($LASTEXITCODE -ne 0) { throw "pg_dump failed - is Docker running (docker compose up -d)?" }
docker cp "stylefind-db:/tmp/$file" (Join-Path $dir $file)
docker exec stylefind-db rm "/tmp/$file"
$size = [math]::Round((Get-Item (Join-Path $dir $file)).Length / 1MB, 1)
Write-Host "Backup written: backups\$file ($size MB)" -ForegroundColor Green
