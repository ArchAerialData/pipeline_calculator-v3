#!/usr/bin/env bash
set -euo pipefail

# Build a macOS .app using PyInstaller.
# NOTE: Code signing/notarization not included here (CI signs separately).

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv"

# Optional overrides:
#   BUNDLE_ID=com.yourorg.pipelinecalculator
#   APP_DISPLAY_NAME="Pipeline Calculator v4"
#   PIPELINE_CALCULATOR_BUILD_IMPL=new|legacy  (default: new)
# Default to a reverse-DNS style bundle identifier so the generated Info.plist is valid.
# Override by exporting BUNDLE_ID=... in your environment/CI.
BUNDLE_ID="${BUNDLE_ID:-com.archaerial.pipelinecalculator}"

BUILD_IMPL="${PIPELINE_CALCULATOR_BUILD_IMPL:-new}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing venv at ${VENV_DIR}."
  echo "Run: bash scripts/macos/setup_macos.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

cd "${REPO_DIR}"

python - <<'PY'
import sys
try:
    import tkinter as tk
except Exception as e:
    print("ERROR: tkinter is not available in this environment.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    print("Fix: run `bash scripts/macos/setup_macos.sh` to install Homebrew python + Tk bindings.", file=sys.stderr)
    raise SystemExit(2)

ver = float(getattr(tk, "TkVersion", 0.0))
if ver < 8.6:
    print(f"ERROR: Tcl/Tk {ver} detected. Builds made with Tk 8.5 are known to crash on macOS 26.", file=sys.stderr)
    print("Fix: ensure your venv was created with Homebrew python@3.11 + python-tk@3.11.", file=sys.stderr)
    raise SystemExit(2)
print(f"Tk OK (TkVersion={ver})")
PY

ENTRY_SCRIPT=""
case "${BUILD_IMPL}" in
  legacy)
    ENTRY_SCRIPT="${REPO_DIR}/src/pipeline_calculator_v3.py"
    ;;
  new|modular|package)
    ENTRY_SCRIPT="${REPO_DIR}/src/pipeline_calculator_entry.py"
    ;;
  *)
    echo "Unknown PIPELINE_CALCULATOR_BUILD_IMPL=${BUILD_IMPL}. Use 'legacy' or 'new'."
    exit 1
    ;;
esac

echo "Build impl: ${BUILD_IMPL}"
echo "Entry script: ${ENTRY_SCRIPT}"

ICON_ARGS=()
if [[ -f "icon.icns" ]]; then
  ICON_ARGS+=(--icon "${REPO_DIR}/icon.icns")
fi

BUNDLE_ID_ARGS=()
if [[ -n "${BUNDLE_ID}" ]]; then
  BUNDLE_ID_ARGS+=(--osx-bundle-identifier "${BUNDLE_ID}")
fi

ADD_DATA_ARGS=()
if [[ -f "${REPO_DIR}/README.md" ]]; then
  ADD_DATA_ARGS+=(--add-data "${REPO_DIR}/README.md:.")
fi
if [[ -f "${REPO_DIR}/icon.icns" ]]; then
  ADD_DATA_ARGS+=(--add-data "${REPO_DIR}/icon.icns:.")
fi
if [[ -f "${REPO_DIR}/icon.ico" ]]; then
  ADD_DATA_ARGS+=(--add-data "${REPO_DIR}/icon.ico:.")
fi

rm -rf build dist
VERSION="$(python src/pipeline_calculator/versioning.py --output build/version.json)"
APP_DISPLAY_NAME="${APP_DISPLAY_NAME:-Pipeline Calculator v${VERSION}}"

# Bash 3.2 (macOS default) + `set -u` errors on empty array expansions.
# Use the `${arr[@]+"${arr[@]}"} ` pattern so empty arrays expand safely.
pyinstaller --noconfirm --clean \
  --name "${APP_DISPLAY_NAME}" \
  --add-data "${REPO_DIR}/build/version.json:pipeline_calculator" \
  --windowed \
  --onedir \
  --specpath "build" \
  --paths "${REPO_DIR}/src" \
  ${ICON_ARGS[@]+"${ICON_ARGS[@]}"} \
  ${BUNDLE_ID_ARGS[@]+"${BUNDLE_ID_ARGS[@]}"} \
  ${ADD_DATA_ARGS[@]+"${ADD_DATA_ARGS[@]}"} \
  --hidden-import pipeline_calculator_v3 \
  --hidden-import scipy.spatial \
  --hidden-import scipy._lib.messagestream \
  --hidden-import tkinterdnd2 \
  --hidden-import PIL \
  --additional-hooks-dir "${REPO_DIR}/scripts/pyinstaller_hooks" \
  "${ENTRY_SCRIPT}"

if [[ -d "dist/${APP_DISPLAY_NAME}.app" ]]; then
  rm -rf "dist/Pipeline_Calculator.app"
  mv "dist/${APP_DISPLAY_NAME}.app" "dist/Pipeline_Calculator.app"
fi

APP_PATH="dist/Pipeline_Calculator.app"
if [[ ! -d "${APP_PATH}" ]]; then
  echo "Build failed: ${APP_PATH} not found."
  exit 1
fi

# Stamp numeric bundle fields and retain the full preview version separately.
python - "${APP_PATH}/Contents/Info.plist" <<'PY'
import json, plistlib, sys
from pathlib import Path
metadata = json.loads(Path("build/version.json").read_text())
path = Path(sys.argv[1])
with path.open("rb") as stream:
    info = plistlib.load(stream)
info["CFBundleShortVersionString"] = metadata["numeric_version"]
info["CFBundleVersion"] = metadata["numeric_version"]
info["PipelineCalculatorVersion"] = metadata["version"]
with path.open("wb") as stream:
    plistlib.dump(info, stream)
PY
echo "Build complete: ${APP_PATH}"
