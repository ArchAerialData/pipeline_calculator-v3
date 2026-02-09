$ErrorActionPreference = "Stop"

# Build a self-contained Windows .exe using PyInstaller.
# NOTE: Code signing not included.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$VenvDir = Join-Path $RepoDir ".venv"
$Py = Join-Path $VenvDir "Scripts\python.exe"

if (!(Test-Path $Py)) { throw "Venv not found. Run scripts\\windows\\setup_windows.ps1 first." }

Push-Location $RepoDir
try {
  if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
  if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }

  $IconArgs = @()
  if (Test-Path "icon.ico") { $IconArgs = @("--icon", "icon.ico") }

  & $Py -m PyInstaller --noconfirm --clean --onefile `
    --windowed `
    --name "Pipeline_Calculator_v3" `
    @IconArgs `
    --add-data "README.md;." `
    --add-data "icon.ico;." `
    --add-data "icon.icns;." `
    --hidden-import "scipy.spatial" `
    --hidden-import "scipy._lib.messagestream" `
    --hidden-import "tkinterdnd2" `
    --hidden-import "PIL" `
    --additional-hooks-dir (Join-Path $RepoDir "scripts\\pyinstaller_hooks") `
    "src\\pipeline_calculator_v3.py"

  if (!(Test-Path "dist\\Pipeline_Calculator_v3.exe")) { throw "Build failed: dist\\Pipeline_Calculator_v3.exe not found." }
  Write-Host "Build complete: dist\\Pipeline_Calculator_v3.exe"
}
finally {
  Pop-Location
}

