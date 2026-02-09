$ErrorActionPreference = "Stop"

# Pipeline Calculator v3 Windows setup script
# - Creates repo-local venv at .venv\
# - Installs requirements
#
# Run (PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\windows\setup_windows.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$VenvDir = Join-Path $RepoDir ".venv"

Write-Host "Repo: $RepoDir"

if (!(Test-Path (Join-Path $RepoDir "requirements.txt"))) { throw "requirements.txt not found at repo root." }
if (!(Test-Path (Join-Path $RepoDir "src\pipeline_calculator_v3.py"))) { throw "src\pipeline_calculator_v3.py not found." }

if (!(Test-Path $VenvDir)) {
  python -m venv $VenvDir
}

$Py = Join-Path $VenvDir "Scripts\python.exe"
& $Py -m pip install --upgrade pip wheel setuptools
& $Py -m pip install -r (Join-Path $RepoDir "requirements.txt")
if (Test-Path (Join-Path $RepoDir "requirements-dev.txt")) {
  & $Py -m pip install -r (Join-Path $RepoDir "requirements-dev.txt")
}

& $Py -c "import tkinter; print('tkinter OK')"
& $Py -c "import customtkinter; print('customtkinter OK')"
& $Py -c "import tkinterdnd2; print('tkinterdnd2 OK')"
& $Py -c "import numpy, pandas, scipy, pyproj; print('scientific stack OK')"

Write-Host "Setup complete."
Write-Host "Next: powershell -ExecutionPolicy Bypass -File scripts\windows\run_gui.ps1"
