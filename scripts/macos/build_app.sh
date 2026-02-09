#!/usr/bin/env bash
set -euo pipefail

# Build a macOS .app using PyInstaller.
# NOTE: Code signing/notarization not included here (CI signs separately).

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv"

# Optional overrides:
#   BUNDLE_ID=com.yourorg.pipelinecalculator
#   APP_DISPLAY_NAME="Pipeline Calculator v3"
BUNDLE_ID="${BUNDLE_ID:-}"
APP_DISPLAY_NAME="${APP_DISPLAY_NAME:-Pipeline Calculator v3}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing venv at ${VENV_DIR}."
  echo "Run: bash scripts/macos/setup_macos.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

cd "${REPO_DIR}"

ICON_ARGS=()
if [[ -f "icon.icns" ]]; then
  ICON_ARGS+=(--icon "icon.icns")
fi

BUNDLE_ID_ARGS=()
if [[ -n "${BUNDLE_ID}" ]]; then
  BUNDLE_ID_ARGS+=(--osx-bundle-identifier "${BUNDLE_ID}")
fi

rm -rf build dist

pyinstaller --noconfirm --clean \
  --name "${APP_DISPLAY_NAME}" \
  --windowed \
  --onedir \
  --specpath "build" \
  "${ICON_ARGS[@]}" \
  "${BUNDLE_ID_ARGS[@]}" \
  --add-data "README.md:." \
  --add-data "icon.icns:." \
  --add-data "icon.ico:." \
  --hidden-import scipy.spatial \
  --hidden-import scipy._lib.messagestream \
  --hidden-import tkinterdnd2 \
  --hidden-import PIL \
  --additional-hooks-dir "${REPO_DIR}/scripts/pyinstaller_hooks" \
  "${REPO_DIR}/src/pipeline_calculator_v3.py"

if [[ -d "dist/${APP_DISPLAY_NAME}.app" ]]; then
  rm -rf "dist/Pipeline_Calculator.app"
  mv "dist/${APP_DISPLAY_NAME}.app" "dist/Pipeline_Calculator.app"
fi

APP_PATH="dist/Pipeline_Calculator.app"
if [[ ! -d "${APP_PATH}" ]]; then
  echo "Build failed: ${APP_PATH} not found."
  exit 1
fi

echo "Build complete: ${APP_PATH}"

