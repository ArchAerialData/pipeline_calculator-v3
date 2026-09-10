$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$VenvDir = Join-Path $RepoDir ".venv"
$Py = Join-Path $VenvDir "Scripts\python.exe"

if (!(Test-Path $Py)) { throw "Venv not found. Run scripts\\windows\\setup_windows.ps1 first." }

& $Py -m compileall (Join-Path $RepoDir "src")
if ($LASTEXITCODE -ne 0) { throw "Python compilation failed ($LASTEXITCODE)" }

& $Py -m pytest
if ($LASTEXITCODE -ne 0) { throw "Tests failed ($LASTEXITCODE); install requirements-dev.txt if pytest is missing." }

Write-Host "OK"
