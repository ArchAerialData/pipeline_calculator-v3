param([string]$OutputRoot = "")
$ErrorActionPreference = "Stop"

# Build a self-contained Windows .exe using PyInstaller.
# NOTE: Code signing not included.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$VenvDir = Join-Path $RepoDir ".venv"
$Py = Join-Path $VenvDir "Scripts\python.exe"
$ArtifactRoot = if ([string]::IsNullOrWhiteSpace($OutputRoot)) { $RepoDir } else { [IO.Path]::GetFullPath($OutputRoot) }
$BuildDir = Join-Path $ArtifactRoot "build"
$DistDir = Join-Path $ArtifactRoot "dist"
$MetadataPath = Join-Path $BuildDir "version.json"

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
$Entry = Join-Path $RepoDir $Entry
$SourceDir = Join-Path $RepoDir "src"
$ReadmePath = Join-Path $RepoDir "README.md"
$WindowsIcon = Join-Path $RepoDir "icon.ico"
$MacIcon = Join-Path $RepoDir "icon.icns"

Write-Host "Build impl: $BuildImpl"
Write-Host "Entry script: $Entry"

Push-Location $RepoDir
try {
  foreach ($TargetDir in @($BuildDir, $DistDir)) {
    $ResolvedTarget = [IO.Path]::GetFullPath($TargetDir)
    $ExpectedParent = [IO.Path]::GetFullPath($ArtifactRoot).TrimEnd('\')
    if ((Split-Path -Parent $ResolvedTarget).TrimEnd('\') -ne $ExpectedParent) { throw "Unsafe build cleanup path: $ResolvedTarget" }
    if (Test-Path -LiteralPath $ResolvedTarget) {
      if ((Get-Item -LiteralPath $ResolvedTarget).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing to clean linked build directory: $ResolvedTarget" }
      Remove-Item -LiteralPath $ResolvedTarget -Recurse -Force
    }
  }

  $Version = & $Py src/pipeline_calculator/versioning.py --output $MetadataPath
  if ($LASTEXITCODE -ne 0) { throw "Version generation failed" }
  $ArtifactName = "Pipeline_Calculator_v$Version"

  $IconArgs = @()
  if (Test-Path $WindowsIcon) { $IconArgs = @("--icon", $WindowsIcon) }

  & $Py -m PyInstaller --noconfirm --clean --onefile `
    --windowed `
    --name $ArtifactName `
    --workpath $BuildDir `
    --distpath $DistDir `
    --specpath $BuildDir `
    @IconArgs `
    --paths $SourceDir `
    --add-data "$MetadataPath;pipeline_calculator" `
    --add-data "$ReadmePath;." `
    --add-data "$WindowsIcon;." `
    --add-data "$MacIcon;." `
    --hidden-import "pipeline_calculator_v3" `
    --hidden-import "scipy.spatial" `
    --hidden-import "scipy._lib.messagestream" `
    --hidden-import "tkinterdnd2" `
    --hidden-import "PIL" `
    --additional-hooks-dir (Join-Path $RepoDir "scripts\\pyinstaller_hooks") `
    $Entry
  if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)" }

  $ArtifactPath = Join-Path $DistDir "$ArtifactName.exe"
  if (!(Test-Path -LiteralPath $ArtifactPath)) { throw "Build failed: $ArtifactPath not found." }
  Write-Host "Build complete: $ArtifactPath"
}
finally {
  Pop-Location
}
