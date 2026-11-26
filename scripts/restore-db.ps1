# Restore a backup made by backup-db.ps1. THIS REPLACES the current database contents.
#   powershell -ExecutionPolicy Bypass -File scripts\restore-db.ps1 -File backups\stylefind_YYYYMMDD-HHMMSS.dump
param([Parameter(Mandatory = $true)][string]$File)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $File)) { throw "Backup file not found: $File" }
$answer = Read-Host "This will REPLACE the stylefind database with $File. Type YES to continue"
if ($answer -ne "YES") { Write-Host "Cancelled."; exit 1 }
$name = Split-Path $File -Leaf
docker cp $File "stylefind-db:/tmp/$name"
# --clean --if-exists drops existing objects first; --single-transaction makes it all-or-nothing
docker exec stylefind-db pg_restore -U postgres -d stylefind --clean --if-exists --single-transaction "/tmp/$name"
if ($LASTEXITCODE -ne 0) { throw "pg_restore failed - nothing was changed (single transaction)" }
docker exec stylefind-db rm "/tmp/$name"
Write-Host "Restored $name" -ForegroundColor Green
