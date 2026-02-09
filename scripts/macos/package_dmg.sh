#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

APP_PATH="${1:-${REPO_DIR}/dist/Pipeline_Calculator.app}"
OUT_DIR="${2:-${REPO_DIR}/dist}"
VOLNAME="${3:-Pipeline Calculator v3}"

if [[ ! -d "${APP_PATH}" ]]; then
  echo "Missing app bundle: ${APP_PATH}"
  exit 1
fi

mkdir -p "${OUT_DIR}"

STAGING_DIR="${OUT_DIR}/dmg_staging"
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

cp -R "${APP_PATH}" "${STAGING_DIR}/"
ln -s /Applications "${STAGING_DIR}/Applications"

DMG_PATH="${OUT_DIR}/Pipeline_Calculator_v3.dmg"
rm -f "${DMG_PATH}"

hdiutil create -volname "${VOLNAME}" -srcfolder "${STAGING_DIR}" -ov -format UDZO "${DMG_PATH}"

rm -rf "${STAGING_DIR}"

echo "DMG created: ${DMG_PATH}"

