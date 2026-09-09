$ErrorActionPreference = "Stop"

# Build a self-contained Windows .exe using PyInstaller.
# NOTE: Code signing not included.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$VenvDir = Join-Path $RepoDir ".venv"
$Py = Join-Path $VenvDir "Scripts\python.exe"

if (!(Test-Path $Py)) { throw "Venv not found. Run scripts\\windows\\setup_windows.ps1 first." }

$BuildImpl = $env:PIPELINE_CALCULATOR_BUILD_IMPL
if ([string]::IsNullOrWhiteSpace($BuildImpl)) { $BuildImpl = "new" }
$BuildImpl = $BuildImpl.ToLowerInvariant()

$Entry = $null
switch ($BuildImpl) {
  "legacy" { $Entry = "src\\pipeline_calculator_v3.py" }
  { $_ -in @("new", "modular", "package") } { $Entry = "src\\pipeline_calculator_entry.py" }
  default { throw "Unknown PIPELINE_CALCULATOR_BUILD_IMPL='$BuildImpl'. Use 'legacy' or 'new'." }
}

Write-Host "Build impl: $BuildImpl"
Write-Host "Entry script: $Entry"

Push-Location $RepoDir
try {
  if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
  if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }

  $Version = & $Py src/pipeline_calculator/versioning.py --output build/version.json
  if ($LASTEXITCODE -ne 0) { throw "Version generation failed" }
  $ArtifactName = "Pipeline_Calculator_v$Version"

  $IconArgs = @()
  if (Test-Path "icon.ico") { $IconArgs = @("--icon", "icon.ico") }

  & $Py -m PyInstaller --noconfirm --clean --onefile `
    --windowed `
    --name $ArtifactName `
    @IconArgs `
    --paths "src" `
    --add-data "build/version.json;pipeline_calculator" `
    --add-data "README.md;." `
    --add-data "icon.ico;." `
    --add-data "icon.icns;." `
    --hidden-import "pipeline_calculator_v3" `
    --hidden-import "scipy.spatial" `
    --hidden-import "scipy._lib.messagestream" `
    --hidden-import "tkinterdnd2" `
    --hidden-import "PIL" `
    --additional-hooks-dir (Join-Path $RepoDir "scripts\\pyinstaller_hooks") `
    $Entry

  if (!(Test-Path "dist\\$ArtifactName.exe")) { throw "Build failed: dist\\$ArtifactName.exe not found." }
  Write-Host "Build complete: dist\\$ArtifactName.exe"
}
finally {
  Pop-Location
}
