#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_PATH="${1:-${REPO_DIR}/dist/Pipeline_Calculator.app}"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }

if [[ ! -d "${APP_PATH}" ]]; then
  echo "Missing app bundle: ${APP_PATH}"
  exit 1
fi

: "${MACOS_CERT_P12:?Missing MACOS_CERT_P12 (base64 .p12)}"
: "${MACOS_CERT_PASSWORD:?Missing MACOS_CERT_PASSWORD}"

WORK_DIR="$(mktemp -d)"
KEYCHAIN="${WORK_DIR}/ci-signing.keychain-db"
KEYCHAIN_PASSWORD="$(
  PY_BIN="$(command -v python3 || true)"
  if [[ -z "${PY_BIN}" ]]; then
    PY_BIN="$(command -v python || true)"
  fi
  if [[ -z "${PY_BIN}" ]]; then
    echo "python not found" >&2
    exit 1
  fi
  "${PY_BIN}" - <<'PY'
import secrets
import string
alphabet = string.ascii_letters + string.digits
print("".join(secrets.choice(alphabet) for _ in range(32)))
PY
)"
CERT_P12="${WORK_DIR}/cert.p12"

cleanup() {
  security delete-keychain "${KEYCHAIN}" >/dev/null 2>&1 || true
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

decode_base64() {
  local input="$1"
  local output="$2"
  if base64 --decode </dev/null >/dev/null 2>&1; then
    printf '%s' "${input}" | base64 --decode > "${output}"
  else
    printf '%s' "${input}" | base64 -D > "${output}"
  fi
}

log "Preparing signing materials..."
decode_base64 "${MACOS_CERT_P12}" "${CERT_P12}"

log "Creating + unlocking temporary keychain..."
security create-keychain -p "${KEYCHAIN_PASSWORD}" "${KEYCHAIN}"
security set-keychain-settings -lut 21600 "${KEYCHAIN}"
security unlock-keychain -p "${KEYCHAIN_PASSWORD}" "${KEYCHAIN}"
security default-keychain -s "${KEYCHAIN}"

log "Importing Developer ID certificate into keychain..."
# -A avoids GUI access prompts (headless CI); keep minimal tool allowlist.
security import "${CERT_P12}" -k "${KEYCHAIN}" -P "${MACOS_CERT_PASSWORD}" -A -T /usr/bin/codesign -T /usr/bin/security
security list-keychain -d user -s "${KEYCHAIN}"

log "Granting codesign key access (partition list)..."
security set-key-partition-list -S apple-tool:,apple: -s -k "${KEYCHAIN_PASSWORD}" "${KEYCHAIN}" >/dev/null

log "Locating Developer ID identity..."
IDENTITY="$(security find-identity -v -p codesigning "${KEYCHAIN}" | awk -F\" '/Developer ID Application/ {print $2; exit}')"
if [[ -z "${IDENTITY}" ]]; then
  echo "Developer ID Application identity not found in keychain."
  exit 1
fi

log "Signing app with identity: ${IDENTITY}"
codesign --force --options runtime --timestamp --sign "${IDENTITY}" --keychain "${KEYCHAIN}" --deep "${APP_PATH}"
codesign --verify --deep --strict --verbose=2 "${APP_PATH}"

log "Packaging DMG..."
bash "${REPO_DIR}/scripts/macos/package_dmg.sh" "${APP_PATH}"
VERSION="$(/usr/libexec/PlistBuddy -c 'Print :PipelineCalculatorVersion' "${APP_PATH}/Contents/Info.plist")"
DMG_PATH="${REPO_DIR}/dist/Pipeline_Calculator_v${VERSION}.dmg"

log "Distribution artifact ready (signed app inside DMG; not notarized): ${DMG_PATH}"
