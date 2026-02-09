$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$VenvDir = Join-Path $RepoDir ".venv"
$Py = Join-Path $VenvDir "Scripts\python.exe"

if (!(Test-Path $Py)) { throw "Venv not found. Run scripts\\windows\\setup_windows.ps1 first." }

& $Py (Join-Path $RepoDir "src\pipeline_calculator_v3.py")

