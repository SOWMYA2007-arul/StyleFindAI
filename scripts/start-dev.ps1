# Starts the three StyleFind services in separate PowerShell windows.
#   powershell -ExecutionPolicy Bypass -File scripts\start-dev.ps1
# Prerequisite: PostgreSQL is running (docker compose up -d) and setup steps in README are done.
$root = Split-Path -Parent $PSScriptRoot

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\ai-service'; .\.venv\Scripts\Activate.ps1; uvicorn app.main:app --host 127.0.0.1 --port 8001"
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; npm run dev"
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"

Write-Host "AI service : http://127.0.0.1:8001/docs"
Write-Host "Backend    : http://localhost:4000/api/health"
Write-Host "Website    : http://localhost:5173  (opens in a few seconds)"
