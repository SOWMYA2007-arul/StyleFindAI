# StyleFind AI - Phase 1 environment check for Windows PowerShell.
# Run from the project folder:   powershell -ExecutionPolicy Bypass -File scripts\check-environment.ps1
# It only READS your system; it installs and changes nothing.

function Show($name, $ok, $detail) {
    $mark = if ($ok) { "[OK]     " } else { "[MISSING]" }
    $color = if ($ok) { "Green" } else { "Yellow" }
    Write-Host ("{0} {1,-22} {2}" -f $mark, $name, $detail) -ForegroundColor $color
}
function Try-Cmd($cmd, $cmdArgs) {
    try { return (& $cmd @cmdArgs 2>&1 | Select-Object -First 1).ToString().Trim() } catch { return $null }
}

Write-Host "`nStyleFind AI - environment check`n" -ForegroundColor Cyan

$py = Try-Cmd "py" @("-3.11", "--version")
if (-not $py) { $py = Try-Cmd "python" @("--version") }
Show "Python 3.11" ($py -match "3\.11") ($(if ($py) { $py } else { "not found - install from python.org (3.11.x)" }))

$node = Try-Cmd "node" @("--version")
Show "Node.js 20+/22" ($node -match "^v(2[0-9])") ($(if ($node) { $node } else { "not found - install Node.js 22 LTS" }))
$npm = Try-Cmd "npm" @("--version")
Show "npm" ([bool]$npm) ($(if ($npm) { $npm } else { "comes with Node.js" }))

$docker = Try-Cmd "docker" @("--version")
Show "Docker" ([bool]$docker) ($(if ($docker) { $docker } else { "optional - easiest way to get PostgreSQL + pgvector" }))
if ($docker) {
    $running = Try-Cmd "docker" @("info", "--format", "{{.ServerVersion}}")
    Show "Docker engine running" ($running -and $running -notmatch "error") ($(if ($running) { $running } else { "start Docker Desktop" }))
}

$psql = Try-Cmd "psql" @("--version")
Show "PostgreSQL client" ([bool]$psql) ($(if ($psql) { $psql } else { "not on PATH (fine if you use Docker)" }))
$pgsvc = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue | Select-Object -First 1
Show "PostgreSQL service" ([bool]$pgsvc) ($(if ($pgsvc) { "$($pgsvc.Name) is $($pgsvc.Status)" } else { "no native service (fine if you use Docker)" }))

$git = Try-Cmd "git" @("--version")
Show "Git" ([bool]$git) ($(if ($git) { $git } else { "optional" }))

$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
Show "RAM" ($ram -ge 8) "$ram GB (16 GB recommended; FashionCLIP needs ~1.5 GB)"
$gpu = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ", "
$nvidia = $gpu -match "NVIDIA"
Show "GPU" $true ("{0} -> {1}" -f $gpu, $(if ($nvidia) { "CUDA possible" } else { "CPU inference (supported)" }))
$free = [math]::Round((Get-PSDrive -Name (Get-Location).Drive.Name).Free / 1GB, 1)
Show "Free disk" ($free -ge 10) "$free GB (need ~6 GB: dataset 0.6 GB, models 2-3 GB, node_modules, venv)"

if (Test-Path ".env") { Show ".env file" $true "present" } else { Show ".env file" $false "run: Copy-Item .env.example .env   then edit passwords" }
Write-Host ""
