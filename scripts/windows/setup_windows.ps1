param([string]$Python = "")
$ErrorActionPreference = "Stop"

# Pipeline Calculator v4 Windows setup script
# - Creates repo-local Python 3.11 venv at .venv\
# - Installs requirements
#
# Run (PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\windows\setup_windows.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$VenvDir = Join-Path $RepoDir ".venv"
$Py = Join-Path $VenvDir "Scripts\python.exe"
$VersionCheck = "import sys; print('Python ' + sys.version.split()[0]); sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)"

Write-Host "Repo: $RepoDir"

if (!(Test-Path (Join-Path $RepoDir "requirements.txt"))) { throw "requirements.txt not found at repo root." }
if (!(Test-Path (Join-Path $RepoDir "src\pipeline_calculator_entry.py"))) { throw "src\pipeline_calculator_entry.py not found." }

if (Test-Path $VenvDir) {
  if (!(Test-Path $Py)) {
    throw "The existing .venv has no Windows Python interpreter. Preserve or rename it, then rerun setup with Python 3.11."
  }
  & $Py -c $VersionCheck
  if ($LASTEXITCODE -ne 0) {
    throw "The existing .venv must use Python 3.11. Preserve or rename it, then rerun setup with -Python <path-to-python-3.11.exe>."
  }
} else {
  $PythonPrefix = @()
  if (![string]::IsNullOrWhiteSpace($Python)) {
    $BootstrapPython = $Python
  } elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $BootstrapPython = "py"
    $PythonPrefix = @("-3.11")
  } elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $BootstrapPython = "python"
  } else {
    throw "Python 3.11 is required. Install it from python.org, then rerun setup with -Python <path-to-python-3.11.exe>."
  }
  & $BootstrapPython @PythonPrefix -c $VersionCheck
  if ($LASTEXITCODE -ne 0) {
    throw "Python 3.11 is required. Install it from python.org or pass -Python <path-to-python-3.11.exe>. No environment was created."
  }
  & $BootstrapPython @PythonPrefix -m venv $VenvDir
  if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed ($LASTEXITCODE)." }
  if (!(Test-Path $Py)) { throw "Virtual environment creation did not produce $Py." }
  & $Py -c $VersionCheck
  if ($LASTEXITCODE -ne 0) { throw "The newly created .venv must use Python 3.11." }
}

& $Py -m pip install --upgrade pip wheel setuptools
if ($LASTEXITCODE -ne 0) { throw "Build tools installation failed ($LASTEXITCODE)." }
& $Py -m pip install -r (Join-Path $RepoDir "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Application dependencies installation failed ($LASTEXITCODE)." }
if (Test-Path (Join-Path $RepoDir "requirements-dev.txt")) {
  & $Py -m pip install -r (Join-Path $RepoDir "requirements-dev.txt")
  if ($LASTEXITCODE -ne 0) { throw "Test dependencies installation failed ($LASTEXITCODE)." }
}

& $Py -c "import tkinter; print('tkinter OK')"
if ($LASTEXITCODE -ne 0) { throw "Tkinter import failed ($LASTEXITCODE)." }
& $Py -c "import customtkinter; print('customtkinter OK')"
if ($LASTEXITCODE -ne 0) { throw "CustomTkinter import failed ($LASTEXITCODE)." }
& $Py -c "import tkinterdnd2; print('tkinterdnd2 OK')"
if ($LASTEXITCODE -ne 0) { throw "TkinterDnD import failed ($LASTEXITCODE)." }
& $Py -c "import numpy, pandas, scipy, pyproj; print('scientific stack OK: NumPy ' + numpy.__version__ + ', SciPy ' + scipy.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Scientific stack import failed ($LASTEXITCODE)." }
& $Py -m pip check
if ($LASTEXITCODE -ne 0) { throw "Installed dependencies are inconsistent ($LASTEXITCODE)." }

Write-Host "Setup complete."
Write-Host "Next: powershell -ExecutionPolicy Bypass -File scripts\windows\run_gui.ps1"
