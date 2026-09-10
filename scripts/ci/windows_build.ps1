$ErrorActionPreference = "Stop"

# CI entrypoint for Windows builds. Keeps workflow YAML minimal.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path

Write-Host "Repo: $RepoDir"

# Reuse the user-facing build script; it expects a venv, so create a venv here.
$VenvDir = Join-Path $RepoDir ".venv"
if (!(Test-Path $VenvDir)) {
  python -m venv $VenvDir
  if ($LASTEXITCODE -ne 0) { throw "Venv creation failed ($LASTEXITCODE)" }
}
$Py = Join-Path $VenvDir "Scripts\python.exe"
& $Py -m pip install --upgrade pip wheel setuptools
if ($LASTEXITCODE -ne 0) { throw "Build tools installation failed ($LASTEXITCODE)" }
& $Py -m pip install -r (Join-Path $RepoDir "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Dependencies installation failed ($LASTEXITCODE)" }
if (Test-Path (Join-Path $RepoDir "requirements-dev.txt")) {
  & $Py -m pip install -r (Join-Path $RepoDir "requirements-dev.txt")
  if ($LASTEXITCODE -ne 0) { throw "Test dependencies installation failed ($LASTEXITCODE)" }
}

& $Py -m py_compile (Join-Path $RepoDir "src\\pipeline_calculator_v3.py")
if ($LASTEXITCODE -ne 0) { throw "Legacy compilation failed ($LASTEXITCODE)" }
if (Test-Path (Join-Path $RepoDir "src\\pipeline_calculator_entry.py")) {
  & $Py -m py_compile (Join-Path $RepoDir "src\\pipeline_calculator_entry.py")
  if ($LASTEXITCODE -ne 0) { throw "Entrypoint compilation failed ($LASTEXITCODE)" }
}
& $Py -m pytest
if ($LASTEXITCODE -ne 0) { throw "Tests failed ($LASTEXITCODE)" }

powershell -ExecutionPolicy Bypass -File (Join-Path $RepoDir "scripts\\windows\\build_exe.ps1")
if ($LASTEXITCODE -ne 0) { throw "Windows build failed ($LASTEXITCODE)" }
