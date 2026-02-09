#!/usr/bin/env bash
set -euo pipefail

# Helper for setting GitHub Actions secrets via the GitHub CLI (`gh`).
#
# This script never prints secret values; it reads them from local files under
# `apple_dev/` (gitignored) and pushes them to GitHub as repository secrets.
#
# Usage:
#   bash scripts/ci/set_github_secrets.sh
#   bash scripts/ci/set_github_secrets.sh owner/repo
#
# Prereqs:
# - `gh` installed and authenticated (`gh auth login`)
# - `apple_dev/` contains the base64 files generated locally

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if ! command -v gh >/dev/null 2>&1; then
  die "GitHub CLI not found. Install it from: https://cli.github.com/"
fi

REPO_SLUG="${1:-}"
if [[ -z "${REPO_SLUG}" ]]; then
  # Try to infer from `origin`.
  if git -C "${REPO_DIR}" remote get-url origin >/dev/null 2>&1; then
    ORIGIN_URL="$(git -C "${REPO_DIR}" remote get-url origin)"
    # Supports:
    # - git@github.com:owner/repo.git
    # - https://github.com/owner/repo.git
    # - https://github.com/owner/repo
    if [[ "${ORIGIN_URL}" == git@github.com:* ]]; then
      REPO_SLUG="${ORIGIN_URL#git@github.com:}"
    elif [[ "${ORIGIN_URL}" == https://github.com/* ]]; then
      REPO_SLUG="${ORIGIN_URL#https://github.com/}"
    fi
    REPO_SLUG="${REPO_SLUG%.git}"
  fi
fi

if [[ -z "${REPO_SLUG}" ]]; then
  die "Could not infer GitHub repo slug. Re-run with: bash scripts/ci/set_github_secrets.sh owner/repo"
fi

echo "Target repo: ${REPO_SLUG}"

P12_B64_FILE="${REPO_DIR}/apple_dev/MACOS_CERT_P12.base64.txt"
if [[ ! -f "${P12_B64_FILE}" ]]; then
  die "Missing ${P12_B64_FILE}. Generate it locally (and keep it gitignored) before running this script."
fi

echo "Setting secret: MACOS_CERT_P12"
gh secret set MACOS_CERT_P12 -R "${REPO_SLUG}" --app actions < "${P12_B64_FILE}"

if [[ -z "${MACOS_CERT_PASSWORD:-}" ]]; then
  # Read from TTY to avoid leaking in shell history; works in an interactive terminal.
  read -r -s -p "Enter MACOS_CERT_PASSWORD (will not echo): " MACOS_CERT_PASSWORD
  echo
fi
if [[ -z "${MACOS_CERT_PASSWORD}" ]]; then
  die "MACOS_CERT_PASSWORD is required."
fi

echo "Setting secret: MACOS_CERT_PASSWORD"
gh secret set MACOS_CERT_PASSWORD -R "${REPO_SLUG}" --app actions --body "${MACOS_CERT_PASSWORD}"

# Optional (future notarization). These secrets are not used by the current workflow.
P8_B64_FILE="${REPO_DIR}/apple_dev/APPLE_API_KEY_P8.base64.txt"
KEY_ID_FILE="${REPO_DIR}/apple_dev/APPLE_API_KEY_ID.txt"
ISSUER_ID_FILE="${REPO_DIR}/apple_dev/APPLE_API_ISSUER_ID.txt"

if [[ -f "${P8_B64_FILE}" && -f "${KEY_ID_FILE}" && -f "${ISSUER_ID_FILE}" ]]; then
  echo "Setting optional secrets (notarization; currently unused): APPLE_API_KEY_P8, APPLE_API_KEY_ID, APPLE_API_ISSUER_ID"
  gh secret set APPLE_API_KEY_P8 -R "${REPO_SLUG}" --app actions < "${P8_B64_FILE}"
  gh secret set APPLE_API_KEY_ID -R "${REPO_SLUG}" --app actions --body "$(cat "${KEY_ID_FILE}")"
  gh secret set APPLE_API_ISSUER_ID -R "${REPO_SLUG}" --app actions --body "$(cat "${ISSUER_ID_FILE}")"
else
  echo "Skipping optional notarization secrets (files not found under apple_dev/)."
fi

echo "Done. You can now run the GitHub Actions workflow; macOS builds on main/tags will require signing secrets."
