# Production-style local start: one Node process serves the built website + API on port 4000,
# the AI service runs on 127.0.0.1:8001 (not exposed). Put HTTPS in front (see docs/DEPLOYMENT.md).
#   powershell -ExecutionPolicy Bypass -File scripts\start-prod.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location "$root\frontend"; npm run build; Pop-Location
Push-Location "$root\backend"; npm run build; Pop-Location
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\ai-service'; .\.venv\Scripts\Activate.ps1; uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1"
Start-Sleep -Seconds 3
$env:NODE_ENV = "production"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; `$env:NODE_ENV='production'; npm start"
Write-Host "Website + API: http://localhost:4000  (health: /api/health)"
