$ErrorActionPreference = "Stop"

# CI entrypoint for Windows builds. Keeps workflow YAML minimal.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path

Write-Host "Repo: $RepoDir"

# Reuse the user-facing build script; it expects a venv, so create a venv here.
$VenvDir = Join-Path $RepoDir ".venv"
if (!(Test-Path $VenvDir)) { python -m venv $VenvDir }
$Py = Join-Path $VenvDir "Scripts\python.exe"
& $Py -m pip install --upgrade pip wheel setuptools
& $Py -m pip install -r (Join-Path $RepoDir "requirements.txt")
if (Test-Path (Join-Path $RepoDir "requirements-dev.txt")) {
  & $Py -m pip install -r (Join-Path $RepoDir "requirements-dev.txt")
}

& $Py -m py_compile (Join-Path $RepoDir "src\\pipeline_calculator_v3.py")
& $Py -m pytest

powershell -ExecutionPolicy Bypass -File (Join-Path $RepoDir "scripts\\windows\\build_exe.ps1")
