#!/usr/bin/env bash
set -euo pipefail

# Notarize + staple a signed DMG using an App Store Connect API key.
#
# Intended flow:
# 1) GitHub Actions builds + code-signs `dist/Pipeline_Calculator_v3.dmg`
# 2) Download the DMG locally
# 3) Run this script to notarize + staple the DMG
#
# Credentials:
# - Provide via env vars, OR via gitignored `apple_dev/` files:
#   - `APPLE_API_KEY_P8.base64.txt` (base64 of the .p8)
#   - `APPLE_API_KEY_ID.txt`
#   - `APPLE_API_ISSUER_ID.txt`
#
# Usage:
#   bash scripts/macos/notarize_dmg.sh dist/Pipeline_Calculator_v3.dmg
#
# Notes:
# - The DMG/app must already be Developer ID signed for notarization to succeed.
# - Notarization requires Apple Developer Program access and network access.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DMG_PATH="${1:-${REPO_DIR}/dist/Pipeline_Calculator_v3.dmg}"
if [[ ! -f "${DMG_PATH}" ]]; then
  echo "DMG not found: ${DMG_PATH}" >&2
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script must be run on macOS." >&2
  exit 1
fi

if ! command -v xcrun >/dev/null 2>&1; then
  echo "xcrun not found. Install Xcode Command Line Tools first." >&2
  exit 1
fi

read_file_trim() {
  local path="$1"
  tr -d '\r\n' < "${path}"
}

APPLE_API_KEY_ID="${APPLE_API_KEY_ID:-}"
APPLE_API_ISSUER_ID="${APPLE_API_ISSUER_ID:-}"
APPLE_API_KEY_P8_BASE64="${APPLE_API_KEY_P8_BASE64:-}"
APPLE_API_KEY_P8_PATH="${APPLE_API_KEY_P8_PATH:-}"

if [[ -z "${APPLE_API_KEY_ID}" && -f "${REPO_DIR}/apple_dev/APPLE_API_KEY_ID.txt" ]]; then
  APPLE_API_KEY_ID="$(read_file_trim "${REPO_DIR}/apple_dev/APPLE_API_KEY_ID.txt")"
fi
if [[ -z "${APPLE_API_ISSUER_ID}" && -f "${REPO_DIR}/apple_dev/APPLE_API_ISSUER_ID.txt" ]]; then
  APPLE_API_ISSUER_ID="$(read_file_trim "${REPO_DIR}/apple_dev/APPLE_API_ISSUER_ID.txt")"
fi
if [[ -z "${APPLE_API_KEY_P8_BASE64}" && -f "${REPO_DIR}/apple_dev/APPLE_API_KEY_P8.base64.txt" ]]; then
  APPLE_API_KEY_P8_BASE64="$(read_file_trim "${REPO_DIR}/apple_dev/APPLE_API_KEY_P8.base64.txt")"
fi

if [[ -z "${APPLE_API_KEY_ID}" ]]; then
  echo "Missing APPLE_API_KEY_ID (env var or apple_dev/APPLE_API_KEY_ID.txt)." >&2
  exit 1
fi
if [[ -z "${APPLE_API_ISSUER_ID}" ]]; then
  echo "Missing APPLE_API_ISSUER_ID (env var or apple_dev/APPLE_API_ISSUER_ID.txt)." >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
cleanup() { rm -rf "${WORK_DIR}"; }
trap cleanup EXIT

KEY_PATH="${WORK_DIR}/AuthKey_${APPLE_API_KEY_ID}.p8"
if [[ -n "${APPLE_API_KEY_P8_PATH}" ]]; then
  if [[ ! -f "${APPLE_API_KEY_P8_PATH}" ]]; then
    echo "APPLE_API_KEY_P8_PATH not found: ${APPLE_API_KEY_P8_PATH}" >&2
    exit 1
  fi
  cp "${APPLE_API_KEY_P8_PATH}" "${KEY_PATH}"
else
  if [[ -z "${APPLE_API_KEY_P8_BASE64}" ]]; then
    echo "Missing APPLE_API_KEY_P8_BASE64 (env var or apple_dev/APPLE_API_KEY_P8.base64.txt)." >&2
    exit 1
  fi
  if base64 --decode </dev/null >/dev/null 2>&1; then
    printf '%s' "${APPLE_API_KEY_P8_BASE64}" | base64 --decode > "${KEY_PATH}"
  else
    printf '%s' "${APPLE_API_KEY_P8_BASE64}" | base64 -D > "${KEY_PATH}"
  fi
fi
chmod 600 "${KEY_PATH}" || true

echo "Submitting DMG to Apple Notary Service (this can take several minutes)..."
xcrun notarytool submit "${DMG_PATH}" \
  --key "${KEY_PATH}" \
  --key-id "${APPLE_API_KEY_ID}" \
  --issuer "${APPLE_API_ISSUER_ID}" \
  --wait \
  --timeout 60m

echo "Stapling notarization ticket to DMG..."
xcrun stapler staple "${DMG_PATH}"

echo "Validating stapled ticket..."
xcrun stapler validate "${DMG_PATH}"

echo "Done: notarized + stapled DMG at ${DMG_PATH}"

